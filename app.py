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

# Danh sách cột đầu vào
EXPECTED_COLUMNS = ['Qgas', 'Qwater', 'Oilrate', 'LiqRate', 'DayOn']

# Tiền xử lý dữ liệu đầu vào
def preprocess_input(df):
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().all():
            df[col] = 0.0  # Nếu toàn bộ cột không có giá trị, điền 0
        else:
            df[col] = df[col].fillna(df[col].mean())  # Điền giá trị trung bình nếu có dữ liệu
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0  

    df[EXPECTED_COLUMNS] = df[EXPECTED_COLUMNS].astype(float)
    return df

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Kiểm tra nếu dữ liệu không có
        if not request.data:
            return jsonify({"error": "No CSV data provided"}), 400

        # Đọc dữ liệu từ Power Automate, có thể chứa ký tự '\n'
        csv_data = request.data.decode('utf-8').strip()
        print("Received CSV Data:\n", csv_data)  # Debug log

        # Kiểm tra nếu dữ liệu trống
        if not csv_data:
            return jsonify({"error": "Empty CSV data received"}), 400

        # Thử đọc CSV từ chuỗi dữ liệu
        try:
            df = pd.read_csv(io.StringIO(csv_data), encoding='utf-8-sig', skip_blank_lines=True)
        except Exception as e:
            return jsonify({"error": f"CSV Parsing Error: {str(e)}"}), 400

        print("Parsed DataFrame:\n", df.head())  # Debug log

        # Tiền xử lý dữ liệu
        df = preprocess_input(df)

        # Kiểm tra xem có đủ cột không
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing columns in input: {missing_columns}"}), 400

        # Dự đoán dữ liệu mới
        try:
            feature_df = df[EXPECTED_COLUMNS]
            predictions = model.predict(feature_df)
            df['Predicted_Qoil'] = predictions
        except Exception as e:
            return jsonify({'error': f'Model prediction error: {str(e)}'}), 500

        # Trả về kết quả dưới dạng JSON (dễ sử dụng hơn so với CSV string)
        response_data = df.to_dict(orient='records')

        return jsonify({
            'message': 'CSV processed successfully',
            'predictions': response_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)