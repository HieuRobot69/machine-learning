"""
=======================================================
  TRAIN MODEL SoC + RUL — NASA Battery Dataset
  Phiên bản 2.0 — Cải tiến toàn diện
=======================================================
Cách dùng:
  pip install pandas numpy scikit-learn joblib xgboost
  python train_from_csv.py

Output:
  soc_model.pkl      — model SoC (GradientBoosting)
  rul_model.pkl      — model RUL (Random Forest)
  soc_scaler.pkl     — scaler SoC
  rul_scaler.pkl     — scaler RUL
  soc_features.pkl   — danh sách features SoC
  rul_features.pkl   — danh sách features RUL
  soc_report.txt     — báo cáo kết quả
=======================================================
"""

import numpy as np
import pandas as pd
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ================================================
# HẰNG SỐ VẬT LÝ PIN 18650
# ================================================
CAP_NOMINAL   = 2.0     # Ah danh định
CAP_EOL       = 1.6     # Ah ngưỡng End-of-Life (80% x 2.0)
TEMP_REF      = 25.0    # °C nhiệt độ tham chiếu
R0            = 0.050   # Ohm điện trở nội ban đầu
dR_PER_CYCLE  = 0.0003  # Ohm/cycle tốc độ lão hóa
TRAIN_BATS    = ['B0005','B0006','B0007']
TEST_BAT      = 'B0018'

print("=" * 60)
print("  TRAIN SoC + RUL MODEL — NASA Battery v2.0")
print("=" * 60)

# ================================================
# BƯỚC 1: ĐỌC DỮ LIỆU
# ================================================
print("\n[1/6] Đọc dữ liệu...")
df = pd.read_csv('nasa_battery_raw.csv')
df = df.sort_values(['battery','cycle_idx','time']).reset_index(drop=True)
print(f"      {len(df):,} dòng, {df['battery'].nunique()} pin, {df['cycle_idx'].nunique()} cycles")

# ================================================
# BƯỚC 2: TẠO FEATURES VẬT LÝ
# ================================================
print("\n[2/6] Feature engineering...")

d = df.copy()

# --- Điện trở nội theo lão hóa ---
d['R_est'] = R0 + d['cycle_idx'] * dR_PER_CYCLE

# --- OCV: loại bỏ sụt áp do dòng ---
d['voltage_ocv'] = (d['voltage'] - d['current'] * d['R_est']).clip(2.0, 4.5)

# --- Hệ số nhiệt độ (ảnh hưởng đến dung lượng thực) ---
d['temp_factor'] = 1.0
d.loc[d['temperature'] > 30, 'temp_factor'] = \
    (1.0 - 0.005 * (d.loc[d['temperature'] > 30, 'temperature'] - 30)).clip(0.7, 1.0)
d.loc[d['temperature'] < 20, 'temp_factor'] = \
    (1.0 - 0.008 * (20 - d.loc[d['temperature'] < 20, 'temperature'])).clip(0.7, 1.0)

# --- Dung lượng thực tế sau lão hóa ---
d['cap_actual'] = (CAP_NOMINAL * (1 - 0.0025 * d['cycle_idx'])).clip(1.0, CAP_NOMINAL)
d['cap_norm']   = d['cap_actual'] / CAP_NOMINAL

# --- State of Health (SoH) ---
d['soh'] = d['cap_actual'] / CAP_NOMINAL * 100  # %

# --- Features điện ---
d['power']        = d['voltage'] * np.abs(d['current'])
d['voltage_sq']   = d['voltage_ocv'] ** 2
d['temp_delta']   = d['temperature'] - d['ambient_temp']
d['v_x_temp']     = d['voltage_ocv'] * d['temperature']
d['energy_rate']  = d['power'] * d['temp_factor']  # công suất hiệu dụng

# --- Đạo hàm (tốc độ thay đổi) ---
grp = d.groupby(['battery','cycle_idx'])
d['dV_dt'] = grp['voltage_ocv'].transform(lambda x: x.diff().fillna(0))
d['dI_dt'] = grp['current'].transform(lambda x: x.diff().fillna(0))
d['dT_dt'] = grp['temperature'].transform(lambda x: x.diff().fillna(0))

# --- Rolling features ---
d['ocv_roll_mean']    = grp['voltage_ocv'].transform(lambda x: x.rolling(10,min_periods=1).mean())
d['ocv_roll_std']     = grp['voltage_ocv'].transform(lambda x: x.rolling(10,min_periods=1).std().fillna(0))
d['current_roll_std'] = grp['current'].transform(lambda x: x.rolling(10,min_periods=1).std().fillna(0))
d['temp_roll_mean']   = grp['temperature'].transform(lambda x: x.rolling(10,min_periods=1).mean())
d['power_roll_mean']  = grp['power'].transform(lambda x: x.rolling(10,min_periods=1).mean())

# --- Target SoC điều chỉnh nhiệt ---
d['SoC_adj'] = (d['SoC'] * d['temp_factor']).clip(0, 100)

# ================================================
# BƯỚC 3: TRAIN MODEL SoC
# ================================================
print("\n[3/6] Train model SoC (GradientBoosting)...")

SOC_FEATURES = [
    'voltage_ocv', 'current', 'temperature', 'ambient_temp',
    'R_est', 'temp_factor', 'cap_norm', 'soh',
    'power', 'voltage_sq', 'temp_delta', 'v_x_temp', 'energy_rate',
    'dV_dt', 'dI_dt', 'dT_dt',
    'ocv_roll_mean', 'ocv_roll_std', 'current_roll_std',
    'temp_roll_mean', 'power_roll_mean',
]

d_soc = d.dropna(subset=SOC_FEATURES + ['SoC_adj'])
train_soc = d_soc[d_soc['battery'].isin(TRAIN_BATS)]
test_soc  = d_soc[d_soc['battery'] == TEST_BAT]

soc_scaler = RobustScaler()
X_train_soc = soc_scaler.fit_transform(train_soc[SOC_FEATURES].values)
X_test_soc  = soc_scaler.transform(test_soc[SOC_FEATURES].values)
y_train_soc = train_soc['SoC_adj'].values
y_test_soc  = test_soc['SoC_adj'].values

soc_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.08, max_depth=6,
    subsample=0.8, min_samples_leaf=5, random_state=42
)
soc_model.fit(X_train_soc, y_train_soc)

y_pred_soc = np.clip(soc_model.predict(X_test_soc), 0, 100)
r2_soc   = r2_score(y_test_soc, y_pred_soc)
rmse_soc = np.sqrt(mean_squared_error(y_test_soc, y_pred_soc))
mae_soc  = mean_absolute_error(y_test_soc, y_pred_soc)
print(f"      SoC  R²={r2_soc:.4f}  RMSE={rmse_soc:.3f}%  MAE={mae_soc:.3f}%")

# ================================================
# BƯỚC 4: TÍNH RUL VÀ TRAIN MODEL RUL
# ================================================
print("\n[4/6] Tính RUL và train model RUL...")

# RUL = số cycle còn lại trước khi pin chết (SoH < 80%)
# Tính capacity theo cycle cho từng pin
cycle_cap = df.groupby(['battery','cycle_idx'])['capacity_ah'].mean().reset_index()
cycle_cap.columns = ['battery','cycle_idx','cap_mean']

rul_rows = []
for bat in cycle_cap['battery'].unique():
    bat_df = cycle_cap[cycle_cap['battery']==bat].sort_values('cycle_idx')
    eol_cycles = bat_df[bat_df['cap_mean'] < CAP_EOL]['cycle_idx']
    if len(eol_cycles) > 0:
        eol = eol_cycles.iloc[0]
    else:
        eol = bat_df['cycle_idx'].max() + 20  # chưa chết
    for _, row in bat_df.iterrows():
        rul = max(0, eol - row['cycle_idx'])
        rul_rows.append({
            'battery': bat,
            'cycle_idx': row['cycle_idx'],
            'cap_mean': row['cap_mean'],
            'soh_cycle': row['cap_mean'] / CAP_NOMINAL * 100,
            'rul': rul
        })

rul_df = pd.DataFrame(rul_rows)

# Thêm thống kê theo cycle
cycle_stats = d.groupby(['battery','cycle_idx']).agg(
    v_mean   = ('voltage_ocv','mean'),
    v_std    = ('voltage_ocv','std'),
    i_mean   = ('current','mean'),
    t_mean   = ('temperature','mean'),
    t_max    = ('temperature','max'),
    p_mean   = ('power','mean'),
    r_mean   = ('R_est','mean'),
    dv_mean  = ('dV_dt','mean'),
).reset_index()

rul_full = rul_df.merge(cycle_stats, on=['battery','cycle_idx'], how='left').dropna()

RUL_FEATURES = [
    'cycle_idx','cap_mean','soh_cycle',
    'v_mean','v_std','i_mean','t_mean','t_max',
    'p_mean','r_mean','dv_mean',
]

train_rul = rul_full[rul_full['battery'].isin(TRAIN_BATS)]
test_rul  = rul_full[rul_full['battery'] == TEST_BAT]

rul_scaler = MinMaxScaler()
X_train_rul = rul_scaler.fit_transform(train_rul[RUL_FEATURES].values)
X_test_rul  = rul_scaler.transform(test_rul[RUL_FEATURES].values)
y_train_rul = train_rul['rul'].values
y_test_rul  = test_rul['rul'].values

rul_model = RandomForestRegressor(
    n_estimators=150, max_depth=15,
    min_samples_leaf=3, random_state=42, n_jobs=-1
)
rul_model.fit(X_train_rul, y_train_rul)

y_pred_rul = np.clip(rul_model.predict(X_test_rul), 0, None)
r2_rul   = r2_score(y_test_rul, y_pred_rul)
rmse_rul = np.sqrt(mean_squared_error(y_test_rul, y_pred_rul))
mae_rul  = mean_absolute_error(y_test_rul, y_pred_rul)
print(f"      RUL  R²={r2_rul:.4f}  RMSE={rmse_rul:.1f} cycles  MAE={mae_rul:.1f} cycles")

# ================================================
# BƯỚC 5: LƯU MODEL
# ================================================
print("\n[5/6] Lưu model...")
joblib.dump(soc_model,   'soc_model.pkl')
joblib.dump(rul_model,   'rul_model.pkl')
joblib.dump(soc_scaler,  'soc_scaler.pkl')
joblib.dump(rul_scaler,  'rul_scaler.pkl')
joblib.dump(SOC_FEATURES,'soc_features.pkl')
joblib.dump(RUL_FEATURES,'rul_features.pkl')

# ================================================
# BƯỚC 6: BÁO CÁO
# ================================================
print("\n[6/6] Viết báo cáo...")

feat_imp_soc = sorted(zip(SOC_FEATURES, soc_model.feature_importances_), key=lambda x: -x[1])
feat_imp_rul = sorted(zip(RUL_FEATURES, rul_model.feature_importances_), key=lambda x: -x[1])

report = f"""
SoC + RUL Model Training Report v2.0
======================================
Train: {TRAIN_BATS}  |  Test: {TEST_BAT}

SoC Model (GradientBoosting):
  R²   = {r2_soc:.4f}
  RMSE = {rmse_soc:.4f}%
  MAE  = {mae_soc:.4f}%
  Features ({len(SOC_FEATURES)}): {', '.join(SOC_FEATURES)}

RUL Model (RandomForest):
  R²   = {r2_rul:.4f}
  RMSE = {rmse_rul:.2f} cycles
  MAE  = {mae_rul:.2f} cycles
  Features ({len(RUL_FEATURES)}): {', '.join(RUL_FEATURES)}

Top 5 SoC Features:
{chr(10).join(f'  {f:<22} {v:.4f}' for f,v in feat_imp_soc[:5])}

Top 5 RUL Features:
{chr(10).join(f'  {f:<22} {v:.4f}' for f,v in feat_imp_rul[:5])}
"""
with open('soc_report.txt','w',encoding='utf-8') as f:
    f.write(report)

print(report)
print("✅ Xong! Chạy: python app.py\n")