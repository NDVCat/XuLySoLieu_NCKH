from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

app = Flask(__name__)

# Load mô hình và imputer đã huấn luyện
model = joblib.load('LinearRegression_model.pkl')
imputer = joblib.load('imputer.pkl')

# Danh sách cột mong đợi
expected_columns = ['DayOn', 'Qoil', 'Qgas', 'Qwater', 'GOR', 'ChokeSize', 'Press_WH', 'Oilrate', 'LiqRate', 'GasRate', 'UniqueId', 'Date']

# Hàm xử lý dữ liệu đầu vào
def preprocess_input(data):
    if isinstance(data, dict):  # Nếu là dictionary, chuyển thành DataFrame
        df = pd.DataFrame([data])
    elif isinstance(data, list):  # Nếu là danh sách, chuyển thành DataFrame
        df = pd.DataFrame(data)
    else:
        return None, "Invalid input format"

    # Chỉnh sửa kiểu dữ liệu cho từng cột
    df['UniqueId'] = df['UniqueId'].astype(str)
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOn'] = df['DayOn'].astype(int)
    df['Method'] = df['Method'].astype(str)
    df['Qoil'] = pd.to_numeric(df['Qoil'], errors='coerce')
    df['Qgas'] = pd.to_numeric(df['Qgas'], errors='coerce')
    df['Qwater'] = pd.to_numeric(df['Qwater'], errors='coerce')
    df['GOR'] = pd.to_numeric(df['GOR'], errors='coerce')
    df['ChokeSize'] = pd.to_numeric(df['ChokeSize'], errors='coerce')
    df['Press_WH'] = pd.to_numeric(df['Press_WH'], errors='coerce')
    df['Oilrate'] = pd.to_numeric(df['Oilrate'], errors='coerce')
    df['LiqRate'] = pd.to_numeric(df['LiqRate'], errors='coerce')
    df['GasRate'] = pd.to_numeric(df['GasRate'], errors='coerce')

    # Kiểm tra nếu cột có tất cả giá trị là NaN thì điền 0
    for col in df.columns:
        if df[col].isnull().all():
            df[col].fillna(0, inplace=True)

    # Điền 'Unknown' cho các cột kiểu object
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna('Unknown', inplace=True)

    # Điền giá trị thiếu bằng mô hình đã huấn luyện
    df_imputed = imputer.transform(df)
    df_filled = pd.DataFrame(df_imputed, columns=expected_columns)

    return df_filled, None

@app.route('/fill_missing_and_generate', methods=['POST'])
def fill_missing_and_generate():
    try:
        data = request.get_json()
        
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
            new_entry['Qoil'] = model.predict([[new_entry['DayOn']]])[0]  # Dự đoán giá trị mới
            new_entry['GOR'] += np.random.normal(0, 5)
            new_entry['Press_WH'] += np.random.normal(0, 2)
            new_entry['LiqRate'] += np.random.normal(0, 1)
            new_entry['Date'] = pd.to_datetime(last_row['Date']) + relativedelta(months=i)
            new_data.append(new_entry)
        
        # Định dạng kết quả thành bảng JSON theo thứ tự cột mong muốn
        response = {
            'filled_data': filled_data[expected_columns].to_dict(orient='records')[0],
            'generated_data': pd.DataFrame(new_data, columns=expected_columns).to_dict(orient='records')
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)