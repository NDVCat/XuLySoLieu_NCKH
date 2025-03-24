from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import os
import datetime

app = Flask(__name__)

# Tải mô hình
model_path = 'LinearRegression_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")

model = joblib.load(model_path)

# Danh sách cột đã được dùng để huấn luyện mô hình
expected_columns = ['DayOn', 'Qoil', 'Qgas', 'Qwater', 'GOR', 'ChokeSize', 
                    'Press_WH', 'Oilrate', 'LiqRate', 'GasRate']

# Hàm chuyển đổi ngày từ số (Excel) sang dạng datetime
def convert_excel_date(excel_date):
    try:
        return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(excel_date))).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None

# Hàm chuyển đổi an toàn với giá trị mặc định
def safe_convert(value, dtype, default):
    try:
        return dtype(value)
    except (ValueError, TypeError):
        return default

# Xử lý dữ liệu đầu vào từ Power Automate
def preprocess_input(data):
    if isinstance(data, dict):  # Nếu là dictionary, chuyển thành DataFrame
        df = pd.DataFrame([data])
    elif isinstance(data, list):  # Nếu là danh sách, chuyển thành DataFrame
        df = pd.DataFrame(data)
    else:
        return None, "Invalid input format"

    print("✅ Dữ liệu ban đầu nhận được:", df.to_dict(orient='records'))

    # Nếu có Date, xử lý định dạng ngày
    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(lambda x: convert_excel_date(x) if isinstance(x, (int, float)) else x)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Nếu có DayOn, chuyển thành số nguyên
    if 'DayOn' in df.columns:
        df['DayOn'] = df['DayOn'].apply(lambda x: safe_convert(x, int, 0))

    # Thêm các cột bị thiếu dựa vào danh sách expected_columns
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0.0  # Điền giá trị mặc định nếu cột bị thiếu

    # Chuyển đổi tất cả các cột số về dạng float để tránh lỗi mô hình
    for col in expected_columns:
        df[col] = df[col].astype(float)

    print("✅ Dữ liệu sau khi xử lý:", df.to_dict(orient='records'))
    return df, None

@app.route('/fill_missing_and_generate', methods=['POST'])
def fill_missing_and_generate():
    try:
        data = request.get_json()
        if not data:
            print("🚨 Lỗi: Không nhận được dữ liệu")
            return jsonify({'error': 'No data received'}), 400

        print("📥 Nhận dữ liệu:", data)

        # Xử lý dữ liệu đầu vào
        filled_data, error = preprocess_input(data)
        if error:
            print("🚨 Lỗi xử lý dữ liệu:", error)
            return jsonify({'error': error}), 400

        # Dự đoán dữ liệu tiếp theo
        new_data = []
        last_row = filled_data.iloc[0]
        for i in range(1, 6):
            new_entry = last_row.copy()
            new_entry['DayOn'] += i
            if 'Date' in last_row and pd.notnull(last_row['Date']):
                new_entry['Date'] = (last_row['Date'] + relativedelta(months=i)).strftime('%Y-%m-%d')

            # Dự đoán với model (Fix lỗi "X does not have valid feature names")
            feature_df = pd.DataFrame([new_entry[expected_columns]])  # Chuyển về DataFrame có tên cột
            try:
                new_entry['Qoil'] = model.predict(feature_df)[0]
                print(f"🔮 Dự đoán {i}: Qoil = {new_entry['Qoil']}")
            except Exception as e:
                print(f"🚨 Lỗi dự đoán với model: {e}")
                return jsonify({'error': f'Model prediction error: {str(e)}'}), 500

            # Tạo nhiễu ngẫu nhiên
            if 'GOR' in new_entry:
                new_entry['GOR'] += np.random.normal(0, 5)
            if 'Press_WH' in new_entry:
                new_entry['Press_WH'] += np.random.normal(0, 2)
            if 'LiqRate' in new_entry:
                new_entry['LiqRate'] += np.random.normal(0, 1)

            new_data.append(new_entry)

        # Fix lỗi "NaTType does not support timetuple"
        for row in new_data:
            if 'Date' in row and pd.isnull(row['Date']):
                row['Date'] = None  # Chuyển NaT thành None để tránh lỗi JSON

        # Trả về JSON kết quả
        response = {
            'filled_data': filled_data.to_dict(orient='records')[0],
            'generated_data': pd.DataFrame(new_data).to_dict(orient='records')
        }

        print("✅ Kết quả trả về:", response)

        return jsonify(response)
    except Exception as e:
        print(f"🚨 Lỗi hệ thống: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Mặc định 10000 nếu không có PORT
    print(f"🚀 Server chạy trên port {port}")
    app.run(host='0.0.0.0', port=port)