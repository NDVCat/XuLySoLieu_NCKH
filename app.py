from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import os
import datetime
import base64

app = Flask(__name__)

# Tải mô hình
MODEL_PATH = "LinearRegression_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")

model = joblib.load(MODEL_PATH)

# Danh sách cột đầu vào của mô hình
EXPECTED_COLUMNS = ['DayOn', 'Qoil', 'Qgas', 'Qwater', 'GOR', 'ChokeSize', 
                    'Press_WH', 'Oilrate', 'LiqRate', 'GasRate']

# Thư mục lưu file upload
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hàm chuyển đổi ngày từ số (Excel) sang datetime
def convert_excel_date(excel_date):
    try:
        return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(excel_date))).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None

# Xử lý file upload từ Base64
def process_uploaded_base64(filename, filecontent):
    try:
        # Giải mã file từ Base64
        file_bytes = base64.b64decode(filecontent)

        # Lưu file
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Đọc file vào DataFrame
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            return None, "Unsupported file format"

        return df, None
    except Exception as e:
        return None, str(e)

# Tiền xử lý dữ liệu đầu vào
def preprocess_input(df):
    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(lambda x: convert_excel_date(x) if isinstance(x, (int, float)) else x)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    if 'DayOn' in df.columns:
        df['DayOn'] = df['DayOn'].astype(int)

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0  

    df[EXPECTED_COLUMNS] = df[EXPECTED_COLUMNS].astype(float)

    return df, None

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        data = request.get_json()
        filename = data.get("filename", "uploaded_file.csv")
        filecontent = data.get("filecontent", "")

        if not filecontent:
            return jsonify({"error": "No file content provided"}), 400

        # Xử lý file Base64
        df, error = process_uploaded_base64(filename, filecontent)
        if error:
            return jsonify({"error": error}), 400

        # Tiền xử lý dữ liệu
        filled_data, error = preprocess_input(df)
        if error:
            return jsonify({"error": error}), 400

        # Dự đoán dữ liệu mới
        new_data = []
        last_row = filled_data.iloc[0]
        for i in range(1, 6):
            new_entry = last_row.copy()
            new_entry['DayOn'] += i
            if 'Date' in last_row and pd.notnull(last_row['Date']):
                new_entry['Date'] = (last_row['Date'] + relativedelta(months=i)).strftime('%Y-%m-%d')

            try:
                feature_df = pd.DataFrame([new_entry[EXPECTED_COLUMNS]])  
                new_entry['Qoil'] = model.predict(feature_df)[0]
            except Exception as e:
                return jsonify({'error': f'Model prediction error: {str(e)}'}), 500

            # Thêm nhiễu ngẫu nhiên
            new_entry['GOR'] += np.random.normal(0, 5)
            new_entry['Press_WH'] += np.random.normal(0, 2)
            new_entry['LiqRate'] += np.random.normal(0, 1)

            new_data.append(new_entry)

        # Xử lý NaN
        for row in new_data:
            if 'Date' in row and pd.isnull(row['Date']):
                row['Date'] = None

        # Xuất file CSV để lưu vào SharePoint từ Power Automate
        output_file = os.path.join(UPLOAD_FOLDER, "predicted_data.csv")
        pd.DataFrame(new_data).to_csv(output_file, index=False)

        response = {
            'message': 'File processed successfully',
            'filled_data': filled_data.to_dict(orient='records'),
            'generated_data': pd.DataFrame(new_data).to_dict(orient='records'),
            'output_file': output_file  # Đường dẫn để tải file CSV từ Power Automate
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
