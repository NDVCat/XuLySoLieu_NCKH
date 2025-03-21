from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load mô hình đã lưu
model = joblib.load('LinearRegression_model.pkl')

# Hàm xử lý dữ liệu đầu vào
def preprocess_input(data):
    df = pd.DataFrame([data])
    
    # Chuyển đổi kiểu dữ liệu tương tự như khi huấn luyện
    df['DayOn'] = df['DayOn'].astype(int)
    df['Qoil'] = pd.to_numeric(df['Qoil'], errors='coerce')
    df['GOR'] = pd.to_numeric(df['GOR'], errors='coerce')
    df['Press_WH'] = pd.to_numeric(df['Press_WH'], errors='coerce')
    df['LiqRate'] = pd.to_numeric(df['LiqRate'], errors='coerce')
    
    # Điền giá trị thiếu
    imputer = SimpleImputer(strategy='mean')
    df_imputed = imputer.fit_transform(df)
    
    return df_imputed

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)
        return jsonify({'predicted_oilrate': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)