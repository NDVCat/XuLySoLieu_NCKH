from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import os
import io

app = Flask(__name__)

# Tải mô hình
MODEL_PATH = "LinearRegression_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")

model = joblib.load(MODEL_PATH)

# Danh sách cột đầu vào của mô hình
EXPECTED_COLUMNS = ['DayOn', 'Qoil', 'Qgas', 'Qwater', 'GOR', 'ChokeSize', 
                    'Press_WH', 'Oilrate', 'LiqRate', 'GasRate']

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
        if not request.data:
            return jsonify({"error": "No CSV data provided"}), 400
        
        csv_data = request.data.decode('utf-8')  # Đọc dữ liệu từ request body
        df = pd.read_csv(io.StringIO(csv_data))
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

        # Chuyển đổi dữ liệu sang CSV string
        output_csv = pd.DataFrame(new_data).to_csv(index=False)
        
        response = {
            'message': 'CSV processed successfully',
            'generated_csv': output_csv  # Trả về CSV dạng chuỗi để Power Automate sử dụng
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)