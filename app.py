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

# Thư mục lưu file upload
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hàm chuyển đổi ngày từ số (Excel) sang dạng datetime
def convert_excel_date(excel_date):
    try:
        return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(excel_date))).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None

# Hàm đọc và xử lý file
def process_uploaded_file(file):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)  # Lưu file tạm thời

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            return None, "Unsupported file format"

        return df, None
    except Exception as e:
        return None, str(e)

# Hàm xử lý dữ liệu đầu vào
def preprocess_input(df):
    print("✅ Dữ liệu ban đầu nhận được:", df.to_dict(orient='records'))

    # Nếu có Date, xử lý định dạng ngày
    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(lambda x: convert_excel_date(x) if isinstance(x, (int, float)) else x)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Nếu có DayOn, chuyển thành số nguyên
    if 'DayOn' in df.columns:
        df['DayOn'] = df['DayOn'].astype(int)

    # Thêm các cột bị thiếu
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0.0  

    # Chuyển đổi dữ liệu về dạng float
    df[expected_columns] = df[expected_columns].astype(float)

    print("✅ Dữ liệu sau khi xử lý:", df.to_dict(orient='records'))
    return df, None

@app.route('/upload', methods=['POST'])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Đọc và xử lý file
    df, error = process_uploaded_file(file)
    if error:
        return jsonify({"error": error}), 400

    # Tiền xử lý dữ liệu
    filled_data, error = preprocess_input(df)
    if error:
        return jsonify({"error": error}), 400

    # Dự đoán dữ liệu tiếp theo
    new_data = []
    last_row = filled_data.iloc[0]
    for i in range(1, 6):
        new_entry = last_row.copy()
        new_entry['DayOn'] += i
        if 'Date' in last_row and pd.notnull(last_row['Date']):
            new_entry['Date'] = (last_row['Date'] + relativedelta(months=i)).strftime('%Y-%m-%d')

        # Dự đoán
        feature_df = pd.DataFrame([new_entry[expected_columns]])  
        try:
            new_entry['Qoil'] = model.predict(feature_df)[0]
        except Exception as e:
            return jsonify({'error': f'Model prediction error: {str(e)}'}), 500

        # Thêm nhiễu ngẫu nhiên
        new_entry['GOR'] += np.random.normal(0, 5)
        new_entry['Press_WH'] += np.random.normal(0, 2)
        new_entry['LiqRate'] += np.random.normal(0, 1)

        new_data.append(new_entry)

    # Fix lỗi "NaTType does not support timetuple"
    for row in new_data:
        if 'Date' in row and pd.isnull(row['Date']):
            row['Date'] = None

    response = {
        'filled_data': filled_data.to_dict(orient='records'),
        'generated_data': pd.DataFrame(new_data).to_dict(orient='records')
    }

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)