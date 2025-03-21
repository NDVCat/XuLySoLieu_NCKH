import joblib
from flask import Flask, request, jsonify
import pandas as pd
import os

# Tải mô hình đã huấn luyện
try:
    model = joblib.load('LinearRegression_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            data = request.get_json(force=True)
        else:  # Nếu là GET, lấy dữ liệu từ query parameters
            data = request.args.to_dict()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Kiểm tra và ép kiểu dữ liệu
        required_columns = ['DayOn', 'Qoil', 'GOR', 'Press_WH', 'LiqRate']  # Thay đổi theo các cột mà mô hình yêu cầu
        for column in required_columns:
            if column not in data:
                return jsonify({'error': f'Missing required input: {column}'}), 400

        try:
            # Chuyển đổi dữ liệu thành float
            data = {key: float(value) for key, value in data.items() if value is not None and value != ""}
        except ValueError:
            return jsonify({'error': 'Invalid input format. All values must be numeric'}), 400

        print("📥 Dữ liệu nhận được:", data)  # Debugging

        # Chuyển đổi dữ liệu thành DataFrame
        input_data = pd.DataFrame([data])

        # Thực hiện dự đoán
        prediction = model.predict(input_data)

        return jsonify({'Predicted_Oilrate': prediction[0]})
    except Exception as e:
        print("Lỗi:", str(e))  # Debugging
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render cung cấp biến PORT
    print(f"Running on port {port}")  # Debug
    app.run(host='0.0.0.0', port=port)