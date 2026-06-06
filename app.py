"""
SoC + RUL Prediction — Web App v2.0
Chạy train_from_csv.py trước, rồi: python app.py
"""
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib, os
import warnings
warnings.filterwarnings('ignore')

app     = Flask(__name__)
BASE    = os.path.dirname(os.path.abspath(__file__))

soc_model    = joblib.load(os.path.join(BASE,'soc_model.pkl'))
rul_model    = joblib.load(os.path.join(BASE,'rul_model.pkl'))
soc_scaler   = joblib.load(os.path.join(BASE,'soc_scaler.pkl'))
rul_scaler   = joblib.load(os.path.join(BASE,'rul_scaler.pkl'))
SOC_FEATURES = joblib.load(os.path.join(BASE,'soc_features.pkl'))
RUL_FEATURES = joblib.load(os.path.join(BASE,'rul_features.pkl'))

R0           = 0.050
dR_PER_CYCLE = 0.0003
CAP_NOMINAL  = 2.0
CAP_EOL      = 1.6
print("✅ Model sẵn sàng! http://localhost:5000")

HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BatteryOS — SoC Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0e1a;
  --surface: #111827;
  --surface2: #1a2234;
  --border: rgba(255,255,255,0.07);
  --accent: #00e5ff;
  --accent2: #7c3aed;
  --green: #10b981;
  --yellow: #f59e0b;
  --orange: #f97316;
  --red: #ef4444;
  --text: #f1f5f9;
  --muted: #64748b;
  --mono: 'Space Mono', monospace;
  --sans: 'DM Sans', sans-serif;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--text); font-family:var(--sans); min-height:100vh; overflow-x:hidden; }

/* Header */
.header {
  display:flex; align-items:center; justify-content:space-between;
  padding:18px 32px; border-bottom:1px solid var(--border);
  background:rgba(10,14,26,0.95); backdrop-filter:blur(12px);
  position:sticky; top:0; z-index:100;
}
.logo { font-family:var(--mono); font-size:16px; font-weight:700; color:var(--accent); letter-spacing:2px; }
.logo span { color:var(--muted); font-weight:400; }
.badge-live { display:flex; align-items:center; gap:6px; font-size:11px; color:var(--green); font-family:var(--mono); }
.dot-live { width:6px; height:6px; border-radius:50%; background:var(--green); animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.4)} }

/* Layout */
.main { max-width:1200px; margin:0 auto; padding:28px 24px; display:grid; grid-template-columns:380px 1fr; gap:20px; }

/* Cards */
.card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:16px; padding:22px;
}
.card-title {
  font-family:var(--mono); font-size:10px; letter-spacing:2px;
  color:var(--muted); text-transform:uppercase; margin-bottom:20px;
  display:flex; align-items:center; gap:8px;
}
.card-title::before { content:''; display:block; width:3px; height:14px; background:var(--accent); border-radius:2px; }

/* Presets */
.presets { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:22px; }
.preset { padding:5px 12px; border:1px solid var(--border); border-radius:6px;
  font-size:11px; font-family:var(--mono); cursor:pointer; background:transparent;
  color:var(--muted); transition:all .15s; }
.preset:hover { border-color:var(--accent); color:var(--accent); background:rgba(0,229,255,0.05); }

/* Sliders */
.field { margin-bottom:20px; }
.field-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px; }
.field-label { font-size:12px; color:var(--muted); font-family:var(--mono); letter-spacing:.5px; }
.field-val { font-family:var(--mono); font-size:20px; font-weight:700; color:var(--text); }
.field-unit { font-size:11px; color:var(--muted); margin-left:3px; }
.field-hint { font-size:10px; color:rgba(100,116,139,0.6); margin-top:4px; }
input[type=range] {
  width:100%; -webkit-appearance:none; height:3px;
  background:var(--surface2); border-radius:2px; outline:none; cursor:pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance:none; width:16px; height:16px; border-radius:50%;
  background:var(--accent); border:2px solid var(--bg); cursor:pointer;
  box-shadow:0 0 8px rgba(0,229,255,0.5); transition:transform .1s;
}
input[type=range]::-webkit-slider-thumb:hover { transform:scale(1.2); }

/* Right panel */
.right-panel { display:flex; flex-direction:column; gap:16px; }

/* SOC Display */
.soc-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:16px; padding:24px;
  display:flex; align-items:center; gap:28px;
}
.soc-ring-wrap { position:relative; flex-shrink:0; }
.soc-ring-wrap svg { transform:rotate(-90deg); }
.soc-ring-center {
  position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
  text-align:center;
}
.soc-pct { font-family:var(--mono); font-size:28px; font-weight:700; line-height:1; }
.soc-label { font-size:9px; color:var(--muted); letter-spacing:1px; margin-top:2px; font-family:var(--mono); }
.soc-info { flex:1; }
.soc-status { font-size:22px; font-weight:600; margin-bottom:4px; }
.soc-desc { font-size:12px; color:var(--muted); line-height:1.6; }
.alert-box {
  margin-top:12px; padding:10px 14px; border-radius:8px;
  font-size:12px; font-family:var(--mono); display:none;
  border-left:3px solid;
}

/* Stats grid */
.stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.stat-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:16px;
}
.stat-label { font-size:10px; color:var(--muted); font-family:var(--mono); letter-spacing:1px; text-transform:uppercase; }
.stat-value { font-family:var(--mono); font-size:22px; font-weight:700; margin:6px 0 2px; }
.stat-sub { font-size:11px; color:var(--muted); }
.stat-bar { height:3px; border-radius:2px; margin-top:10px; background:var(--surface2); overflow:hidden; }
.stat-bar-fill { height:100%; border-radius:2px; transition:width .5s ease; }

/* RUL Card */
.rul-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:16px; padding:22px;
}
.rul-timeline { display:flex; gap:3px; margin-top:14px; }
.rul-seg { height:8px; flex:1; border-radius:2px; background:var(--surface2); transition:background .3s; }

/* Factor breakdown */
.factors { margin-top:14px; }
.factor-row {
  display:flex; justify-content:space-between; align-items:center;
  padding:9px 0; border-bottom:1px solid var(--border); font-size:12px;
}
.factor-row:last-child { border-bottom:none; }
.factor-name { color:var(--muted); font-family:var(--mono); font-size:11px; }
.factor-val { font-weight:600; font-family:var(--mono); font-size:12px; }
.factor-bar-wrap { width:80px; height:3px; background:var(--surface2); border-radius:2px; margin-left:12px; }
.factor-bar-inner { height:100%; border-radius:2px; background:var(--accent); transition:width .4s; }

/* Warnings panel */
.warnings { display:flex; flex-direction:column; gap:8px; }
.warn-item {
  padding:10px 14px; border-radius:8px; font-size:12px;
  display:flex; align-items:center; gap:10px; font-family:var(--mono);
  border:1px solid transparent;
}
.warn-ok   { background:rgba(16,185,129,0.08); border-color:rgba(16,185,129,0.2); color:var(--green); }
.warn-warn { background:rgba(245,158,11,0.08); border-color:rgba(245,158,11,0.2); color:var(--yellow); }
.warn-danger { background:rgba(239,68,68,0.08); border-color:rgba(239,68,68,0.2); color:var(--red); }
.warn-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; background:currentColor; }

@media (max-width:900px) { .main { grid-template-columns:1fr; } }
</style>
</head>
<body>

<div class="header">
  <div class="logo">BATTERY<span>OS</span></div>
  <div class="badge-live"><div class="dot-live"></div>LIVE PREDICTION</div>
</div>

<div class="main">
  <!-- LEFT: INPUT -->
  <div>
    <div class="card">
      <div class="card-title">Input Parameters</div>
      <div class="presets">
        <button class="preset" onclick="sp(4.15,-2.0,25,1)">NEW FULL</button>
        <button class="preset" onclick="sp(3.70,-2.0,28,50)">MID LIFE</button>
        <button class="preset" onclick="sp(3.20,-1.9,35,100)">LOW SOC</button>
        <button class="preset" onclick="sp(2.70,-1.8,40,160)">AGED</button>
        <button class="preset" onclick="sp(4.10,0,24,1)">IDLE FULL</button>
        <button class="preset" onclick="sp(3.50,-2.0,45,80)">HOT</button>
      </div>

      <div class="field">
        <div class="field-header">
          <span class="field-label">VOLTAGE</span>
          <span><span class="field-val" id="v-disp">3.70</span><span class="field-unit">V</span></span>
        </div>
        <input type="range" id="voltage" min="1.7" max="4.3" step="0.01" value="3.7" oninput="go()">
        <div class="field-hint">Điện áp đo được — cao = pin đầy | thấp = pin cạn</div>
      </div>

      <div class="field">
        <div class="field-header">
          <span class="field-label">CURRENT</span>
          <span><span class="field-val" id="i-disp">-2.00</span><span class="field-unit">A</span></span>
        </div>
        <input type="range" id="current" min="-2.1" max="0.5" step="0.01" value="-2.0" oninput="go()">
        <div class="field-hint">Âm = đang phóng điện | dương = đang sạc | 0 = nghỉ</div>
      </div>

      <div class="field">
        <div class="field-header">
          <span class="field-label">TEMPERATURE</span>
          <span><span class="field-val" id="t-disp">28.0</span><span class="field-unit">°C</span></span>
        </div>
        <input type="range" id="temp" min="5" max="55" step="0.5" value="28" oninput="go()">
        <div class="field-hint">Nóng/lạnh quá đều làm dung lượng thực giảm</div>
      </div>

      <div class="field">
        <div class="field-header">
          <span class="field-label">CYCLE COUNT</span>
          <span><span class="field-val" id="c-disp">1</span><span class="field-unit">cycles</span></span>
        </div>
        <input type="range" id="cycle" min="1" max="200" step="1" value="1" oninput="go()">
        <div class="field-hint">Số lần sạc/xả — càng nhiều, pin càng lão hóa</div>
      </div>
    </div>

    <!-- Warnings -->
    <div class="card" style="margin-top:16px;">
      <div class="card-title">System Warnings</div>
      <div class="warnings" id="warnings"></div>
    </div>
  </div>

  <!-- RIGHT: RESULTS -->
  <div class="right-panel">

    <!-- SOC Ring -->
    <div class="soc-card">
      <div class="soc-ring-wrap">
        <svg width="120" height="120" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="50" fill="none" stroke="#1a2234" stroke-width="10"/>
          <circle id="ring-track" cx="60" cy="60" r="50" fill="none" stroke="#00e5ff"
            stroke-width="10" stroke-linecap="round"
            stroke-dasharray="314" stroke-dashoffset="157" style="transition:all .5s ease"/>
        </svg>
        <div class="soc-ring-center">
          <div class="soc-pct" id="ring-pct" style="color:#00e5ff">--%</div>
          <div class="soc-label">SOC</div>
        </div>
      </div>
      <div class="soc-info">
        <div class="soc-status" id="soc-status">--</div>
        <div class="soc-desc" id="soc-desc">Đang tính toán...</div>
        <div class="alert-box" id="alert-box"></div>
      </div>
    </div>

    <!-- Stats 2x2 -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">OCV Voltage</div>
        <div class="stat-value" id="s-ocv" style="color:#00e5ff">--</div>
        <div class="stat-sub">Sau bù sụt áp nội trở</div>
        <div class="stat-bar"><div class="stat-bar-fill" id="bar-ocv" style="width:50%;background:#00e5ff"></div></div>
      </div>
      <div class="stat-card">
        <div class="stat-label">State of Health</div>
        <div class="stat-value" id="s-soh" style="color:#10b981">--</div>
        <div class="stat-sub">Dung lượng còn lại / danh định</div>
        <div class="stat-bar"><div class="stat-bar-fill" id="bar-soh" style="width:80%;background:#10b981"></div></div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Internal Resistance</div>
        <div class="stat-value" id="s-r" style="color:#f59e0b">--</div>
        <div class="stat-sub">Tăng theo chu kỳ sử dụng</div>
        <div class="stat-bar"><div class="stat-bar-fill" id="bar-r" style="width:30%;background:#f59e0b"></div></div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Temp Factor</div>
        <div class="stat-value" id="s-tf" style="color:#a78bfa">--</div>
        <div class="stat-sub">Hiệu suất nhiệt độ thực tế</div>
        <div class="stat-bar"><div class="stat-bar-fill" id="bar-tf" style="width:90%;background:#a78bfa"></div></div>
      </div>
    </div>

    <!-- RUL -->
    <div class="rul-card">
      <div class="card-title">Remaining Useful Life (RUL)</div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;">
        <div>
          <span style="font-family:var(--mono);font-size:36px;font-weight:700;" id="rul-val">--</span>
          <span style="font-size:13px;color:var(--muted);margin-left:6px;">cycles remaining</span>
        </div>
        <div style="text-align:right;">
          <div style="font-family:var(--mono);font-size:13px;" id="rul-km">--</div>
          <div style="font-size:11px;color:var(--muted);">ước tính</div>
        </div>
      </div>
      <div class="rul-timeline" id="rul-timeline"></div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--muted);font-family:var(--mono);margin-top:6px;">
        <span>Hiện tại</span><span>EOL (80% cap)</span>
      </div>
      <div style="margin-top:14px;font-size:12px;color:var(--muted);" id="rul-desc"></div>
    </div>

    <!-- Factor breakdown -->
    <div class="card">
      <div class="card-title">Factor Analysis</div>
      <div class="factors" id="factors"></div>
    </div>

  </div>
</div>

<script>
var timer = null;
function go() {
  var v = +document.getElementById('voltage').value;
  var i = +document.getElementById('current').value;
  var t = +document.getElementById('temp').value;
  var c = +document.getElementById('cycle').value;
  document.getElementById('v-disp').textContent = v.toFixed(2);
  document.getElementById('i-disp').textContent = i.toFixed(2);
  document.getElementById('t-disp').textContent = t.toFixed(1);
  document.getElementById('c-disp').textContent = c;
  clearTimeout(timer);
  timer = setTimeout(function(){ predict(v,i,t,c); }, 100);
}
function sp(v,i,t,c) {
  document.getElementById('voltage').value=v;
  document.getElementById('current').value=i;
  document.getElementById('temp').value=t;
  document.getElementById('cycle').value=c;
  go();
}
function predict(v,i,t,c) {
  fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({voltage:v,current:i,temperature:t,cycle:c})})
  .then(r=>r.json()).then(d=>render(d,v,i,t,c));
}
function render(d,v,i,t,c) {
  var soc = d.soc;
  var rul = d.rul;
  var soh = d.soh;
  var ocv = d.ocv;
  var r   = d.r_est;
  var tf  = d.temp_factor;

  // Ring
  var circ = 314;
  var offset = circ - (soc/100)*circ;
  var ringEl = document.getElementById('ring-track');
  ringEl.style.strokeDashoffset = offset;
  var col = soc>=60?'#10b981':soc>=30?'#f59e0b':soc>=10?'#f97316':'#ef4444';
  ringEl.style.stroke = col;
  document.getElementById('ring-pct').textContent = soc.toFixed(1)+'%';
  document.getElementById('ring-pct').style.color = col;

  // Status
  var status = soc>=60?'Tốt':soc>=30?'Trung bình':soc>=10?'Sắp hết':'Nguy hiểm';
  var desc = soc>=60?'Pin đủ dùng, không cần lo lắng.':
             soc>=30?'Nên sạc pin trong thời gian tới.':
             soc>=10?'Sạc pin ngay để tránh mất dữ liệu.':
             'Pin sắp hết hoàn toàn, cắm sạc gấp!';
  document.getElementById('soc-status').textContent = status;
  document.getElementById('soc-status').style.color = col;
  document.getElementById('soc-desc').textContent = desc;

  // Alert box
  var ab = document.getElementById('alert-box');
  if(t>45){
    ab.style.display='block'; ab.style.borderColor='#ef4444'; ab.style.background='rgba(239,68,68,0.08)'; ab.style.color='#ef4444';
    ab.textContent='⚠ NHIỆT ĐỘ NGUY HIỂM — Pin có thể phồng hoặc cháy!';
  } else if(t>38){
    ab.style.display='block'; ab.style.borderColor='#f97316'; ab.style.background='rgba(249,115,22,0.08)'; ab.style.color='#f97316';
    ab.textContent='! Pin đang nóng — hạn chế sử dụng nặng.';
  } else if(t<10){
    ab.style.display='block'; ab.style.borderColor='#7c3aed'; ab.style.background='rgba(124,58,237,0.08)'; ab.style.color='#a78bfa';
    ab.textContent='! Nhiệt độ quá thấp — dung lượng giảm mạnh.';
  } else { ab.style.display='none'; }

  // Stats
  document.getElementById('s-ocv').textContent = ocv.toFixed(3)+' V';
  document.getElementById('s-soh').textContent = soh.toFixed(1)+'%';
  document.getElementById('s-r').textContent   = (r*1000).toFixed(1)+' mΩ';
  document.getElementById('s-tf').textContent  = (tf*100).toFixed(1)+'%';
  document.getElementById('bar-ocv').style.width = ((ocv-2.5)/2*100).toFixed(0)+'%';
  document.getElementById('bar-soh').style.width = soh+'%';
  document.getElementById('bar-r').style.width   = Math.min(100,(r-0.05)/0.1*100).toFixed(0)+'%';
  document.getElementById('bar-tf').style.width  = (tf*100).toFixed(0)+'%';

  // RUL
  document.getElementById('rul-val').textContent = Math.round(rul);
  var rulCol = rul>50?'#10b981':rul>20?'#f59e0b':'#ef4444';
  document.getElementById('rul-val').style.color = rulCol;
  document.getElementById('rul-km').textContent = 'SoH còn '+soh.toFixed(1)+'%';
  var totalSeg=30, filledSeg=Math.round((rul/200)*totalSeg);
  var tl=document.getElementById('rul-timeline'); tl.innerHTML='';
  for(var s=0;s<totalSeg;s++){
    var seg=document.createElement('div'); seg.className='rul-seg';
    if(s<filledSeg) seg.style.background=s/totalSeg<0.3?'#ef4444':s/totalSeg<0.6?'#f59e0b':'#10b981';
    tl.appendChild(seg);
  }
  var rulDesc = rul>60?'Pin còn khoẻ, tuổi thọ tốt.':
                rul>30?'Pin ổn định, tiếp tục theo dõi.':
                rul>10?'Pin bắt đầu lão hóa, chú ý theo dõi.':
                rul>3?'Pin gần cuối vòng đời, chuẩn bị thay.':
                'Pin cần thay sớm — nguy cơ hỏng đột ngột.';
  document.getElementById('rul-desc').textContent = rulDesc;

  // Factors
  var R0=0.05, dR=0.0003;
  var Ri = R0+c*dR;
  var vDrop = Math.abs(i)*Ri;
  var capLoss = ((1-soh/100)*100).toFixed(1);
  var tempEff = t>30?'-'+((1-tf)*100).toFixed(1)+'% (nóng)':
                t<20?'-'+((1-tf)*100).toFixed(1)+'% (lạnh)':'±0% (lý tưởng)';
  var factors = [
    ['OCV', ocv.toFixed(3)+' V', 'voltage chính xác sau bù', Math.max(0,(ocv-2.5)/2)],
    ['NHIỆT ĐỘ', t.toFixed(1)+'°C → '+tempEff, 'ảnh hưởng dung lượng thực', tf],
    ['LÃO HÓA', 'Cycle '+c+' → mất '+capLoss+'% cap', 'degradation tích lũy', soh/100],
    ['SỤT ÁP', i.toFixed(2)+'A × '+(Ri*1000).toFixed(0)+'mΩ = -'+vDrop.toFixed(3)+'V', 'V đo vs OCV', Math.max(0,1-vDrop/0.2)],
  ];
  var fEl=document.getElementById('factors'); fEl.innerHTML='';
  factors.forEach(function(f){
    fEl.innerHTML+=`<div class="factor-row">
      <span class="factor-name">${f[0]}</span>
      <span class="factor-val">${f[1]}</span>
      <div class="factor-bar-wrap"><div class="factor-bar-inner" style="width:${(f[3]*100).toFixed(0)}%"></div></div>
    </div>`;
  });

  // Warnings
  var warns = [];
  if(soc<10) warns.push({lvl:'danger',msg:'SOC CỰC THẤP — sạc ngay lập tức!'});
  else if(soc<20) warns.push({lvl:'warn',msg:'SoC thấp — nên sạc sớm'});
  else warns.push({lvl:'ok',msg:'SoC bình thường'});
  if(t>45) warns.push({lvl:'danger',msg:'NHIỆT ĐỘ NGUY HIỂM ('+t+'°C) — nguy cơ cháy nổ!'});
  else if(t>38) warns.push({lvl:'warn',msg:'Pin đang nóng ('+t+'°C) — giảm tải'});
  else warns.push({lvl:'ok',msg:'Nhiệt độ bình thường'});
  if(soh<70) warns.push({lvl:'danger',msg:'PIN GẦN HẾT ĐỜI — SoH='+soh.toFixed(0)+'%'});
  else if(soh<80) warns.push({lvl:'warn',msg:'Pin lão hóa đáng kể — SoH='+soh.toFixed(0)+'%'});
  else warns.push({lvl:'ok',msg:'Sức khoẻ pin tốt — SoH='+soh.toFixed(0)+'%'});
  if(rul<10) warns.push({lvl:'danger',msg:'RUL='+Math.round(rul)+' cycles — thay pin sớm!'});
  else if(rul<40) warns.push({lvl:'warn',msg:'RUL='+Math.round(rul)+' cycles — chuẩn bị thay'});
  else warns.push({lvl:'ok',msg:'Tuổi thọ còn '+Math.round(rul)+' cycles'});
  var wEl=document.getElementById('warnings'); wEl.innerHTML='';
  warns.forEach(function(w){
    wEl.innerHTML+=`<div class="warn-item warn-${w.lvl}"><div class="warn-dot"></div>${w.msg}</div>`;
  });
}
go();
</script>
</body>
</html>"""


# Bang tra OCV->SoC thuc te pin 18650 (phi tuyen)
_OCV_TABLE = [
    (4.20,100.0),(4.15,97.0),(4.10,93.0),(4.05,89.0),
    (4.00,84.0),(3.95,79.0),(3.90,74.0),(3.85,68.0),
    (3.80,62.0),(3.75,56.0),(3.70,50.0),(3.65,44.0),
    (3.60,38.0),(3.55,32.0),(3.50,27.0),(3.45,22.0),
    (3.40,18.0),(3.35,14.0),(3.30,11.0),(3.25,8.0),
    (3.20,5.0),(3.10,2.0),(3.00,1.0),(2.50,0.0),
]
def ocv_to_soc(ocv):
    ocv = float(np.clip(ocv, 2.50, 4.20))
    for i in range(len(_OCV_TABLE)-1):
        v_hi,s_hi = _OCV_TABLE[i]
        v_lo,s_lo = _OCV_TABLE[i+1]
        if v_lo <= ocv <= v_hi:
            t = (ocv-v_lo)/(v_hi-v_lo)
            return s_lo + t*(s_hi-s_lo)
    return 0.0

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
  data    = request.json
  voltage = float(data['voltage'])
  current = float(data['current'])
  temp    = float(data['temperature'])
  cycle   = int(data['cycle'])
  ambient = 24.0

  R_est = R0 + cycle * dR_PER_CYCLE
  ocv = float(np.clip(voltage - current * R_est, 2.0, 4.5))

  if temp > 30:   tf = max(0.7, 1.0 - 0.005*(temp-30))
  elif temp < 20: tf = max(0.7, 1.0 - 0.008*(20-temp))
  else:           tf = 1.0

  cap_actual = max(1.0, CAP_NOMINAL*(1-0.0025*cycle))
  cap_norm   = cap_actual / CAP_NOMINAL
  soh        = cap_actual / CAP_NOMINAL * 100

  # Khi dang sac (I>0): dung voltage truc tiep (khong qua OCV)
  # vi khi sac, V_terminal > OCV, OCV chua on dinh
  if current > 0.05:
    soc_base = max(0.0, min(100.0, ocv_to_soc(voltage)))
    soc = float(np.clip(soc_base * tf, 0, 100))
  else:
    X_soc = np.array([[
      ocv, current, temp, ambient, R_est, tf, cap_norm, soh,
      voltage*abs(current), ocv**2, temp-ambient, ocv*temp,
      voltage*abs(current)*tf, 0.0, 0.0, 0.0,
      ocv, 0.0, 0.0, temp, voltage*abs(current)
    ]])
    X_soc_sc = soc_scaler.transform(X_soc)
    soc_ai = float(np.clip(soc_model.predict(X_soc_sc)[0], 0, 100))
    # Khi pin nghi (I~0): BMS rule chinh xac hon model
    if current >= -0.05:
      if voltage >= 4.15:   soc = 100.0
      elif voltage <= 3.2:  soc = 0.0
      else: soc = ocv_to_soc(voltage)
      soc = float(np.clip(soc * tf, 0, 100))
    else:
      # Discharge: dung OCV thuan tuy — da bao gom bu sut ap, khong bi bias
      soc = ocv_to_soc(ocv)
      soc = float(np.clip(soc * tf, 0, 100))

  v_mean = ocv; v_std = 0.05; i_mean = current
  t_max  = temp; p_mean = voltage*abs(current); dv_mean = 0.0
  X_rul = np.array([[
    cycle, cap_actual, soh, v_mean, v_std, i_mean,
    temp, t_max, p_mean, R_est, dv_mean
  ]])
  X_rul_sc = rul_scaler.transform(X_rul)
  rul_pred = float(np.clip(rul_model.predict(X_rul_sc)[0], 0, 300))

  # Nếu pin còn mới (SoH cao), tính RUL từ công thức vật lý thay vì model
  # vì model thiếu data ở vùng pin mới
  # SoH giảm ~0.25%/cycle → cycles_to_eol = (SoH - 80) / 0.25
  rul_formula = max(0, (soh - 80) / 0.25)
  # Blend: pin mới dùng formula, pin già dùng model
  blend = min(1.0, max(0.0, (100 - soh) / 20))  # 0 khi SoH=100, 1 khi SoH=80
  rul = rul_formula * (1 - blend) + rul_pred * blend

  from flask import jsonify
  return jsonify({
    'soc': round(soc,2), 'rul': round(rul,1),
    'soh': round(soh,2), 'ocv': round(ocv,4),
    'r_est': round(R_est,5), 'temp_factor': round(tf,4),
    'cap_ah': round(cap_actual,4),
  })

if __name__=='__main__': app.run(debug=False,port=5000)