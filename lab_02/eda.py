import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _check_outliers_IQR(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def overview_data(path="23_HOMES.csv"):
    os.makedirs("result", exist_ok=True)

    #1. Load data
    data = pd.read_csv(path)

    print("Original Data:")
    print(data)

    print("Data Infomation:")
    print(data.info())

    print("Missing Values:")
    print(data.isnull().sum())

    #2. Handle outliers using 
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    #3. Correlation matrix
    corr_matrix = data.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", mask=mask)
    plt.title("Correlation Matrix Before Preprocessing")
    plt.savefig("result/correlation_BP.png", dpi=300, bbox_inches="tight")
    plt.show()

    #4. Detect outliers using IQR 
    print("Outliers Detected:")
    for column in numeric_cols:
        outliers = _check_outliers_IQR(data, column)
        print(f"{column}:\n{len(outliers)} outliers\n")
        print(outliers)
        print("\n")

    #5. Boxplot before preprocessing
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data)
    plt.xticks(rotation=45)
    plt.title("Boxplot Before Preprocessing")
    plt.savefig("result/boxplot_BP.png", dpi=300, bbox_inches="tight")
    plt.show()

    #6. Histogram before preprocessing
    data.hist(figsize=(12, 8))
    plt.suptitle("Histograms Before Preprocessing")
    plt.savefig("result/histogram_BP.png", dpi=300, bbox_inches="tight")
    plt.show()

    return data

def preprocess_data(data):
 # ==================================================
    # 1. Copy original dataset
    # ==================================================
    data_processed = data.copy()

    # ==================================================
    # 2. Drop List_Price
    # ==================================================
    # Vì List_Price gần giống Selling_Price
    # Correlation quá cao (~0.99)
    # Nếu giữ lại model sẽ học "ăn gian"
    data_processed.drop(columns=["List_Price"], inplace=True)

    # ==================================================
    # 3. Tách X và y
    # ==================================================
    # X = input features
    # y = target cần predict

    X = data_processed.drop(columns=["Selling_Price"])

    y = data_processed[["Selling_Price"]]

    # ==================================================
    # 4. Chia train / validation
    # ==================================================
    # 80% train
    # 20% validation

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ==================================================
    # 5. Scale X bằng StandardScaler
    # ==================================================
    # Đưa feature về mean = 0, std = 1
    # Giúp Linear Regression học ổn định hơn

    x_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)

    X_val_scaled = x_scaler.transform(X_val)

    # Chuyển lại thành DataFrame để dễ nhìn
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    )

    X_val_scaled = pd.DataFrame(
        X_val_scaled,
        columns=X.columns
    )

    # ==================================================
    # 6. Scale y bằng MinMaxScaler
    # ==================================================
    # Đưa giá nhà về khoảng [0,1]
    # Sau này predict xong sẽ inverse_transform()

    y_scaler = MinMaxScaler()

    y_train_scaled = y_scaler.fit_transform(y_train)

    y_val_scaled = y_scaler.transform(y_val)

    # DataFrame cho dễ đọc
    y_train_scaled = pd.DataFrame(
        y_train_scaled,
        columns=["Selling_Price"]
    )

    y_val_scaled = pd.DataFrame(
        y_val_scaled,
        columns=["Selling_Price"]
    )

    # ==================================================
    # 7. In thông tin dataset
    # ==================================================

    print("X_train shape:", X_train_scaled.shape)
    print("X_val shape:", X_val_scaled.shape)

    print("y_train shape:", y_train_scaled.shape)
    print("y_val shape:", y_val_scaled.shape)

    # ==================================================
    # 8.1 Save train dataset
    # ==================================================

    processed_train = pd.concat(
        [X_train_scaled, y_train_scaled],
        axis=1
    )

    processed_train.to_csv(
        "result/23_HOMES_train.csv",
        index=False
    )

    # ==================================================
    # 8.2 Save validation dataset
    # ==================================================

    processed_val = pd.concat(
        [X_val_scaled, y_val_scaled],
        axis=1
    )

    processed_val.to_csv(
        "result/23_HOMES_validation.csv",
        index=False
    )

    print("Train dataset saved.")
    print("Validation dataset saved.")

    # ==================================================
    # 9. Boxplot after preprocessing
    # ==================================================

    plt.figure(figsize=(12,6))

    sns.boxplot(data=processed_train)

    plt.xticks(rotation=45)

    plt.title("Boxplot After Preprocessing")

    plt.savefig(
        "result/boxplot_after.png",
        dpi=300,
        bbox_inches="tight"
    )

    # plt.show()

    # ==================================================
    # 10. Histogram after preprocessing
    # ==================================================

    processed_train.hist(figsize=(12,8))

    plt.suptitle("Histograms After Preprocessing")

    plt.savefig(
        "result/histogram_after.png",
        dpi=300,
        bbox_inches="tight"
    )

    # plt.show()

    # ==================================================
    # 11. Return dữ liệu
    # ==================================================

    return (
        X_train_scaled,
        X_val_scaled,
        y_train_scaled,
        y_val_scaled,
        x_scaler,
        y_scaler
    )

if __name__ == "__main__":
    data = pd.read_csv("23_HOMES.csv")
    X_train_scaled,X_val_scaled,y_train_scaled, y_val_scaled,x_scaler,y_scaler = preprocess_data(data)
    
    print("\n=== X_train_scaled ===")
    print(X_train_scaled)
    
    print("\n=== X_val_scaled ===")
    print(X_val_scaled)
    
    print("\n=== y_train_scaled ===")
    print(y_train_scaled)
    
    print("\n=== y_val_scaled ===")
    print(y_val_scaled)

    print("X scaler:", x_scaler)
    print("y scaler:", y_scaler)