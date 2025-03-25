from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import os

app = Flask(__name__)

# Tải mô hình
MODEL_PATH = "LinearRegression_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")

model = joblib.load(MODEL_PATH)

# Danh sách cột đầu vào của mô hình
EXPECTED_COLUMNS = ['DayOn', 'Qoil', 'Qgas', 'Qwater', 'GOR', 'ChokeSize', 
                    'Press_WH', 'Oilrate', 'LiqRate', 'GasRate']

# Thư mục lưu file tải lên
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

    return df

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Xử lý file CSV
        df = pd.read_csv(file_path)
        df = preprocess_input(df)
        
        # Dự đoán dữ liệu mới
        new_data = []
        last_row = df.iloc[0]
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

        # Xuất file CSV kết quả
        output_file = os.path.join(UPLOAD_FOLDER, "predicted_data.csv")
        pd.DataFrame(new_data).to_csv(output_file, index=False)

        response = {
            'message': 'File processed successfully',
            'file_name': file.filename,
            'generated_data': pd.DataFrame(new_data).to_dict(orient='records'),
            'output_file': output_file  # Đường dẫn để tải file CSV từ Power Automate
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
