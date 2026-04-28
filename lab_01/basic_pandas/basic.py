import pandas as pd
import numpy as np

def test_series():
    a = pd.Series([1, 2, 3, 4, 5])
    print(a)
    b = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
    print(b)

def test_dataframe():
    h = np.array(np.random.rand(5, 5))
    print(h)
    df = pd.DataFrame(h)
    print(df)

    r = np.random.randn(20, 2)
    df2 = pd.DataFrame(r, columns=['A', 'B'])
    print(df2.head())

if "__main__":
    test_dataframe()