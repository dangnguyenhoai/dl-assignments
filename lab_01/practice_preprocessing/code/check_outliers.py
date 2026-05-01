import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Draft code for checking outliers using IQR method and visualizing data
def show_data(path="data/04_CIGARET.csv"):
    data = pd.read_csv(path)
    print(data)

    print(data.info())
    
    # Tính correlation
    corr_matrix = data.corr(numeric_only=True)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", mask=mask)
    plt.savefig("result/correlation.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(data.iloc[0])

    for column in data.columns:
        outliers = check_outliers_IQR(data, column)
        print(f"Outliers in {column}:\n{outliers}\n")
    
    sns.boxplot(data=data)
    plt.xticks(rotation=45)
    plt.savefig("result/boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()
    data.hist(figsize=(12,8))
    plt.savefig("result/histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

def check_outliers_IQR(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

if __name__ == "__main__":
    show_data()