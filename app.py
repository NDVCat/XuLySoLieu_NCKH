from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import io

app = Flask(__name__)

# Tải mô hình
MODEL_PATH = "LinearRegression_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")

model = joblib.load(MODEL_PATH)

# Danh sách cột đầu vào (đã cập nhật theo model mới)
EXPECTED_COLUMNS = ['Qgas', 'Qwater', 'Oilrate', 'LiqRate', 'DayOn']

# Tiền xử lý dữ liệu đầu vào
def preprocess_input(df):
    # Điền giá trị thiếu cho các cột số
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().all():
            df[col] = 0.0  # Nếu toàn bộ cột không có giá trị, điền 0
        else:
            df[col] = df[col].fillna(df[col].mean())  # Điền giá trị trung bình nếu có dữ liệu
    
    # Điền giá trị thiếu cho các cột object
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')
    
    # Đảm bảo các cột EXPECTED_COLUMNS có trong dataframe
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
        
        csv_data = request.data.decode('utf-8')  
        print("Received CSV Data:\n", csv_data)  # Debug log

        # Kiểm tra dữ liệu có phải CSV hợp lệ không
        if not csv_data.strip():
            return jsonify({"error": "Empty CSV data received"}), 400

        # Thử đọc CSV để kiểm tra lỗi
        try:
            df = pd.read_csv(io.StringIO(csv_data))
        except Exception as e:
            return jsonify({"error": f"CSV Parsing Error: {str(e)}"}), 400
        
        print("Parsed DataFrame:\n", df.head())  # Kiểm tra dataframe sau khi đọc

        df = preprocess_input(df)

        # Dự đoán dữ liệu mới
        try:
            feature_df = df[EXPECTED_COLUMNS]  
            print("Feature DataFrame for Model:\n", feature_df.head())  # Debug log
            predictions = model.predict(feature_df)
            df['Predicted_Qoil'] = predictions
        except Exception as e:
            return jsonify({'error': f'Model prediction error: {str(e)}'}), 500

        # Chuyển đổi dữ liệu sang CSV string
        output_csv = df.to_csv(index=False)
        
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
