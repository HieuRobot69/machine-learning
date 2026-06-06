# machine-learning
🔋 BatteryOS: SoC & RUL Prediction System
📌 Giới thiệu dự án (Mô tả)
Dự án BatteryOS ứng dụng Machine Learning để dự đoán Trạng thái sạc (SoC - State of Charge) và Tuổi thọ còn lại (RUL - Remaining Useful Life) của pin Lithium-ion 18650.

Trong các hệ thống Robot và UAV hiện đại, việc quản lý pin chỉ dựa trên điện áp (Voltage) thường dẫn đến sai số lớn do sự sụt áp nội trở và ảnh hưởng của nhiệt độ. Dự án này giải quyết bài toán đó bằng cách sử dụng các mô hình học máy (GradientBoosting và RandomForest) được huấn luyện trên bộ dữ liệu kiểm thử pin của NASA. Hệ thống giúp ước lượng chính xác dung lượng thực tế và đưa ra cảnh báo an toàn trước khi pin đạt đến ngưỡng hư hỏng (End-of-Life).

🚀 Tính năng nổi bật
Dự đoán SoC (Độ chính xác ~94%): Tính toán % pin theo thời gian thực dựa trên dòng phóng/sạc, điện áp, và nhiệt độ môi trường.

Dự đoán RUL: Ước lượng số chu kỳ sạc/xả còn lại trước khi dung lượng pin (SoH) giảm xuống dưới mức an toàn (80%).

Web Dashboard Trực quan: Giao diện điều khiển (phát triển bằng Flask & HTML/CSS/JS) cho phép giả lập các thông số vật lý của pin và theo dõi cảnh báo trực tiếp trên trình duyệt.

Hệ thống cảnh báo an toàn: Tự động phát hiện và cảnh báo các tình trạng bất thường như nhiệt độ quá cao (nguy cơ cháy nổ) hoặc điện áp tụt sâu.

🛠️ Công nghệ & Thư viện sử dụng
Ngôn ngữ: Python, HTML, CSS, JavaScript.

Machine Learning: scikit-learn (GradientBoostingRegressor, RandomForestRegressor).

Xử lý dữ liệu: pandas, numpy.

Triển khai Web: Flask.

📂 Cấu trúc dự án
Plaintext
📦 BatteryOS
 ┣ 📜 app.py               # Source code chạy Web Dashboard dự đoán Real-time
 ┣ 📜 train_from_csv.py    # Source code xử lý dữ liệu và huấn luyện mô hình ML
 ┣ 📜 nasa_battery_raw.csv # Bộ dữ liệu pin gốc từ NASA
 ┣ 📜 soc_report.txt       # File báo cáo kết quả train (R², RMSE, MAE)
 ┣ 📜 README.md            # Tài liệu hướng dẫn dự án
 ┗ 📂 models/              # (Được tạo ra sau khi train)
   ┣ 📜 soc_model.pkl      # Mô hình dự đoán SoC
   ┣ 📜 rul_model.pkl      # Mô hình dự đoán RUL
   ┗ 📜 ... (các file scaler và features)
⚙️ Hướng dẫn cài đặt và sử dụng
Bước 1: Cài đặt các thư viện cần thiết
Mở terminal/command prompt và chạy lệnh sau:

Bash
pip install pandas numpy scikit-learn joblib flask
Bước 2: Huấn luyện mô hình (Training)
Chạy script huấn luyện để tạo ra các file mô hình .pkl từ dữ liệu của NASA:

Bash
python train_from_csv.py
(Quá trình này sẽ mất một chút thời gian. Sau khi xong, hệ thống sẽ tự động sinh ra file soc_report.txt cùng các file model).

Bước 3: Chạy Web Dashboard
Khởi động server Flask để sử dụng giao diện dự đoán:

Bash
python app.py
Mở trình duyệt và truy cập vào địa chỉ: http://localhost:5000 để bắt đầu trải nghiệm hệ thống dự đoán.
