import pandas as pd
from pycaret.classification import *

# 加載數據集
try:
    data_train = pd.read_csv(r"D:\vs code\AIoT_Project\AutoML_Ensemble model optimization\HW4-1\train.csv")
    data_test = pd.read_csv(r"D:\vs code\AIoT_Project\AutoML_Ensemble model optimization\HW4-1\test.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# 特徵處理函數
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

# 特徵處理
data_train = transform_features(data_train)
data_test = transform_features(data_test)

# 清理特徵名稱
data_train.columns = data_train.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
data_test.columns = data_test.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

# 確保類別型特徵為字符串
categorical_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Cabin', 'Lname', 'NamePrefix']
for col in categorical_features:
    data_train[col] = data_train[col].astype(str)
    data_test[col] = data_test[col].astype(str)

# 確認清理後的列名稱
print("清理後的列名稱（訓練集）：", data_train.columns.tolist())
print("清理後的列名稱（測試集）：", data_test.columns.tolist())

# PyCaret 設置
clf1 = setup(data=data_train, 
             target='Survived', 
             categorical_features=categorical_features)

# 創建和調優 LightGBM 模型
lightgbm = create_model('lightgbm', fold=5)
tuned_lightgbm = tune_model(lightgbm, fold=5, optimize='AUC')

# 使用模型進行預測
predictions = predict_model(tuned_lightgbm, data=data_test)

# 提取 PassengerId 和預測結果列
if 'Label' in predictions.columns:
    submission = predictions[['PassengerId', 'Label']]
    submission.rename(columns={'Label': 'Survived'}, inplace=True)
elif 'prediction_label' in predictions.columns:
    submission = predictions[['PassengerId', 'prediction_label']]
    submission.rename(columns={'prediction_label': 'Survived'}, inplace=True)
else:
    raise ValueError("無法找到預測列，請檢查預測結果的列名！")

# 保存預測結果為 CSV 文件
output_path = r"D:\vs code\AIoT_Project\AutoML_Ensemble model optimization\HW4-1\prediction.csv"
submission.to_csv(output_path, index=False)
print(f"預測結果已保存到：{output_path}")
