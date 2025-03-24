from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import os
import datetime

app = Flask(__name__)

model_path = 'LinearRegression_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")

model = joblib.load(model_path)

# Danh sách cột mong đợi
expected_columns = ['DayOn', 'Qoil', 'Qgas', 'Qwater', 'GOR', 'ChokeSize', 
                    'Press_WH', 'Oilrate', 'LiqRate', 'GasRate', 'UniqueId', 'Date']

# Chuyển đổi ngày từ số (Excel) sang định dạng YYYY-MM-DD
def convert_excel_date(excel_date):
    try:
        return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(excel_date))).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None

# Chuyển đổi dữ liệu an toàn
def safe_convert(value, dtype, default):
    try:
        return dtype(value)
    except (ValueError, TypeError):
        return default

# Hàm xử lý dữ liệu đầu vào
def preprocess_input(data):
    if isinstance(data, dict):  # Nếu là dictionary, chuyển thành DataFrame
        df = pd.DataFrame([data])
    elif isinstance(data, list):  # Nếu là danh sách, chuyển thành DataFrame
        df = pd.DataFrame(data)
    else:
        return None, "Invalid input format"

    # Chỉnh sửa kiểu dữ liệu
    df['UniqueId'] = df['UniqueId'].astype(str)
    
    # Xử lý Date (có thể là số từ Excel hoặc dạng chuỗi)
    df['Date'] = df['Date'].apply(lambda x: convert_excel_date(x) if isinstance(x, (int, float)) else x)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df['DayOn'] = df['DayOn'].apply(lambda x: safe_convert(x, int, 0))

    numeric_cols = ['Qoil', 'Qgas', 'Qwater', 'GOR', 'ChokeSize', 
                    'Press_WH', 'Oilrate', 'LiqRate', 'GasRate']
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: safe_convert(x, float, 0.0))

    # Điền giá trị thiếu bằng mô hình đã huấn luyện
    missing_cols = df.columns[df.isnull().any()].tolist()  # Các cột bị thiếu
    for col in missing_cols:
        if col in numeric_cols:  # Chỉ xử lý cột số
            df[col] = df[col].fillna(df[col].median())  # Điền giá trị trung vị

    return df, None

@app.route('/fill_missing_and_generate', methods=['POST'])
def fill_missing_and_generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        # Kiểm tra xem các cột bắt buộc có tồn tại không
        missing_cols = [col for col in expected_columns if col not in data]
        if missing_cols:
            return jsonify({'error': f'Missing required columns: {missing_cols}'}), 400

        # Xử lý dữ liệu đầu vào
        filled_data, error = preprocess_input(data)
        if error:
            return jsonify({'error': error}), 400

        # Dự đoán dữ liệu tiếp theo
        new_data = []
        last_row = filled_data.iloc[0]
        for i in range(1, 6):
            new_entry = last_row.copy()
            new_entry['DayOn'] += i
            new_entry['Date'] = last_row['Date'] + relativedelta(months=i)

            # Dự đoán với model
            feature_array = new_entry.drop(columns=['UniqueId', 'Date']).values.reshape(1, -1)
            new_entry['Qoil'] = model.predict(feature_array)[0]

            # Tạo nhiễu ngẫu nhiên
            new_entry['GOR'] += np.random.normal(0, 5)
            new_entry['Press_WH'] += np.random.normal(0, 2)
            new_entry['LiqRate'] += np.random.normal(0, 1)

            new_data.append(new_entry)

        # Trả về JSON kết quả
        response = {
            'filled_data': filled_data[expected_columns].to_dict(orient='records')[0],
            'generated_data': pd.DataFrame(new_data, columns=expected_columns).to_dict(orient='records')
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Mặc định 10000 nếu không có PORT
    app.run(host='0.0.0.0', port=port)
