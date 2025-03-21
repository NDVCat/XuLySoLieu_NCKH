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
    df['Qgas'] = pd.to_numeric(df['Qgas'], errors='coerce')
    df['Qwater'] = pd.to_numeric(df['Qwater'], errors='coerce')
    df['GOR'] = pd.to_numeric(df['GOR'], errors='coerce')
    df['ChokeSize'] = pd.to_numeric(df['ChokeSize'], errors='coerce')
    df['Press_WH'] = pd.to_numeric(df['Press_WH'], errors='coerce')
    df['Oilrate'] = pd.to_numeric(df['Oilrate'], errors='coerce')
    df['LiqRate'] = pd.to_numeric(df['LiqRate'], errors='coerce')
    df['GasRate'] = pd.to_numeric(df['GasRate'], errors='coerce')
    
    # Điền giá trị 0 cho các cột số
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df[column].fillna(0, inplace=True)
        else:
            df[column].fillna('Unknown', inplace=True)
    
    # Điền giá trị thiếu bằng mô hình ML
    imputer = SimpleImputer(strategy='mean')
    df_imputed = imputer.fit_transform(df)
    
    return df_imputed

@app.route('/fill_missing_and_generate', methods=['POST'])
def fill_missing_and_generate():
    try:
        data = request.get_json()
        
        # Điền giá trị thiếu bằng ML
        processed_data = preprocess_input(data)
        filled_data = pd.DataFrame(processed_data, columns=['DayOn', 'Qoil', 'Qgas', 'Qwater', 'GOR', 'ChokeSize', 'Press_WH', 'Oilrate', 'LiqRate', 'GasRate'])
        
        # Tạo thêm 5 dữ liệu tiếp theo
        new_data = []
        for i in range(1, 6):
            new_entry = {
                'DayOn': filled_data.loc[0, 'DayOn'] + i,
                'Qoil': filled_data.loc[0, 'Qoil'] + np.random.normal(0, 10),
                'GOR': filled_data.loc[0, 'GOR'] + np.random.normal(0, 5),
                'Press_WH': filled_data.loc[0, 'Press_WH'] + np.random.normal(0, 2),
                'LiqRate': filled_data.loc[0, 'LiqRate'] + np.random.normal(0, 1)
            }
            new_data.append(new_entry)
        
        return jsonify({'filled_data': filled_data.to_dict(orient='records')[0], 'generated_data': new_data})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
