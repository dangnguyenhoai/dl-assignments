import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data(path="data/04_CIGARET.csv"):
    data = pd.read_csv(path)
    print(data)

    print(data.info())
    
    # Tính correlation
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.savefig("result/correlation.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(data.iloc[0])

    for column in data.columns:
        outliers = check_outliers(data, column)
        print(f"Outliers in {column}:\n{outliers}\n")
    
    sns.boxplot(data=data)
    plt.xticks(rotation=45)
    plt.savefig("result/boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()
    data.hist(figsize=(12,8))
    plt.savefig("result/histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

def check_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

if __name__ == "__main__":
    preprocess_data()