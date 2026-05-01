import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import RobustScaler


def _check_outliers_IQR(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def get_high_corr_features(data, threshold=0.85):
    corr_matrix = data.corr(numeric_only=True).abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    features_to_drop = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    return features_to_drop

def overview_data(path="data/04_CIGARET.csv"):
    os.makedirs("result", exist_ok=True)

    #1. Load data
    data = pd.read_csv(path)

    print("Original Data:")
    print(data)

    print("Data Infomation:")
    print(data.info())

    print("Missing Values:")
    print(data.isnull().sum())

    #2. Handle outliers using RobustScaler
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

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
    data_processed = data.copy()

    #1. Log transformation for skewed features
    #Use for Tar and CO because these variables have a wider range than Nic
    log_cols = [col for col in data_processed.columns if "Tar" in col or "CO" in col]

    print(f"Columns to apply log transformation: {log_cols}")

    for col in log_cols:
        data_processed[col] = np.log1p(data_processed[col])
    
    #2. Handle multicollinearity
    #Drop highly correlated features 

    high_corr_features = get_high_corr_features(data_processed)

    print(f"Highly correlated features could be dropped: {high_corr_features}")

    #data_processed = data_processed.drop(columns=high_corr_features)

    #3. Robust scaling to handle outliers
    numeric_cols = data_processed.select_dtypes(include=[np.number]).columns
    data_processed[numeric_cols] = RobustScaler().fit_transform(data_processed[numeric_cols])
    
    print("Preprocessed Data:")
    print(data_processed)

    #4. Save processed data
    data_processed.to_csv("result/cigarette_preprocessed.csv", index=False)

    # 5. Boxplot after preprocessing
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=data_processed)
    plt.xticks(rotation=45)
    plt.title("Boxplot After Log Transformation and Robust Scaling")
    plt.savefig("result/boxplot_after.png", dpi=300, bbox_inches="tight")
    plt.show()

    #6. Histogram after preprocessing
    data_processed.hist(figsize=(12, 8))
    plt.suptitle("Histograms After Preprocessing")
    plt.savefig("result/histogram_after.png", dpi=300, bbox_inches="tight")
    plt.show()

    return data_processed

if __name__ == "__main__":
    data = overview_data()
    preprocess_data(data)