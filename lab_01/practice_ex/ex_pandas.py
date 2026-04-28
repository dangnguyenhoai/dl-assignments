import pandas as pd
import numpy as np

def ex_15():
    a = pd.Series([2,4,6,8,10])
    b = pd.Series([1,3,5,7,9])

    print(a)
    print(b)

    print("Sum: ", a + b)
    print("Difference: ", a - b)
    print("Product: ", a * b)
    print("Quotient: ", a / b)


if "__main__":
    ex_15()