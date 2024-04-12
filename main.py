import os
import pandas as pd
from sklearn.model_selection import train_test_split


basePath: str = os.path.dirname(__file__) +"\\Models\\"
csvName = "BigMart_Data.csv"

def readCsv(path: str) -> pd.DataFrame:
    return pd.read_csv(basePath + csvName)
def cleanData(df: pd.DataFrame,showLog :bool = False) -> pd.DataFrame:
    if showLog:
        print("--> Số lượng bản ghi trước khi làm sạch:", df.shape[0])
    missing_values = df.isnull().sum()

    if showLog:
        print("Số lượng bản ghi thiếu giá trị:")
        print(missing_values)

    df.dropna(inplace=True)

    if showLog:
        print("--> Số lượng bản ghi sau khi làm sạch:", df.shape[0])

    return df
def impute_missing_values(df,continuous_vars :[],categorical_vars :[]):

    for var in continuous_vars:
        df[var].fillna(df[var].mean(), inplace=True)

    for var in categorical_vars:
        df[var].fillna(df[var].mode()[0], inplace=True)

    return df
def create_categorical_data(df, var, bins, labels, new_var_name):
    df[new_var_name] = pd.cut(df[var], bins=bins, labels=labels)
    return df

if __name__ == '__main__':
    #1.1
    df = readCsv(basePath + csvName)

    #2.1
    copyName = "pre-processing.csv"
    df_copy = df.copy()
    df_copy.to_csv(basePath + copyName)

    continuous_vars = ['Item_Weight', 'Item_MRP']
    categorical_vars = ['Item_Fat_Content', 'Outlet_Size']

    impute_missing_values(
        df_copy,
        continuous_vars,
        categorical_vars
    )

    mrp_bins = [0, 100, 200, 300, float('inf')]
    mrp_labels = ['Rẻ', 'Trung bình', 'Đắt', 'Rất đắt']

    df_copy = create_categorical_data(df, 'Item_MRP', mrp_bins, mrp_labels, 'Item_MRP_Category')

    weight_bins = [0, 10, 20, 30, float('inf')]
    weight_labels = ['Nhẹ', 'Trung bình', 'Nặng', 'Rất nặng']

    df_copy = create_categorical_data(df, 'Item_Weight', weight_bins, weight_labels, 'Item_Weight_Category')

    train_size = 0.7
    train_df, test_df = train_test_split(
        df_copy,
        train_size=train_size,
        random_state=42
    )

    train_name = "train.csv"
    train_df.to_csv(basePath + train_name, index=False)


    test_df.drop(columns=['Item_Outlet_Sales'])
    test_name="test.csv"
    test_df.to_csv(basePath+test_name, index=False)














