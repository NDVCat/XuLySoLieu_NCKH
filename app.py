from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import os
import requests

app = Flask(__name__)

# Tải mô hình
MODEL_PATH = "LinearRegression_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")

model = joblib.load(MODEL_PATH)

# Danh sách cột đầu vào của mô hình
EXPECTED_COLUMNS = ['DayOn', 'Qoil', 'Qgas', 'Qwater', 'GOR', 'ChokeSize', 
                    'Press_WH', 'Oilrate', 'LiqRate', 'GasRate']

# Thư mục lưu file tải về từ SharePoint
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hàm tải file từ SharePoint
def download_file_from_sharepoint(file_url, save_path):
    try:
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return save_path, None
        else:
            return None, f"Failed to download file: {response.text}"
    except Exception as e:
        return None, str(e)

# Hàm xử lý file
def process_file(file_path):
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            return None, "Unsupported file format"
        
        return df, None
    except Exception as e:
        return None, str(e)

# Tiền xử lý dữ liệu đầu vào
def preprocess_input(df):
    if 'Date' in df.columns:
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
        file_path = data.get("file_path")

        if not file_path:
            return jsonify({"error": "Missing 'file_path' parameter"}), 400

        file_name = os.path.basename(file_path)
        local_file_path = os.path.join(UPLOAD_FOLDER, file_name)

        # Tải file từ SharePoint
        downloaded_path, error = download_file_from_sharepoint(file_path, local_file_path)
        if error:
            return jsonify({"error": error}), 400

        # Xử lý file
        df, error = process_file(downloaded_path)
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
            'file_name': file_name,
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
