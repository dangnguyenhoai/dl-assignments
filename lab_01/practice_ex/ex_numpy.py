import numpy as np

def e_1():
    a = np.array([np.random.randint(0,20,(3,3)) for _ in range(20)])
    print(a)

    if np.all(a != 0):
        print("All elements are different from 0")
    else:
        print("There is at least one element equal to 0")

def e_2():
    a = np.array([np.random.randint(0,20,(3,3)) for _ in range(20)])
    print(a)
    if np.any(a == 0):
        print("There is at least one element equal to 0")
    else:        
        print("All elements are different from 0")

def e_3():
    a = np.array([np.random.randint(0,20,3)])
    b = np.array([np.random.randint(0,20,3)])
    print(a)
    print(b)
    
    print(np.greater(a,b))
    print(np.less(a,b))
    print(np.equal(a,b))
    print(np.greater_equal(a,b))
    print(np.less_equal(a,b))
    
def e_4():
    a = np.array([np.ones(10)])
    b = np.array([np.zeros(10)])
    c = np.array([np.full(10,5.)])

    print(a)
    print(b)
    print(c)
    
def e_5():
    a = np.array(np.random.choice(np.arange(30,70,2),size=10))
    print(a)

def e_6():
    a = np.identity(3)
    print(a)

def e_7():
    a = np.random.uniform(15,56,10)
    print(a)
    print(a[1:9])

def e_8():
    a = np.random.uniform(0,20,20)
    print(a)

    a = [-x if 9 <= x <= 15 else x for x in a ]
    print(a)

def e_9():
    a = np.matrix(np.random.randint(10,22,(3,4)))
    print(a)

def e_10():
    a = np.ones((5,5))  
    a[1:-1,1:-1] = 0
    print(a)

def e_11():
    a = np.diag([x for x in range(1,6)])
    print(a)

def e_12():
    a = np.random.randint(0,20,size =(3,3,3))
    print(a)

    print("Sum by rows:")
    row_sum = np.sum(a, axis=2)
    print(row_sum)

    print("Sum by columns:")
    col_sum = np.sum(a, axis=1)
    print(col_sum)

def e_13():
    a = np.array(np.random.randint(0,10,10))
    b = np.array(np.random.randint(0,10,10))
    print(a)
    print(b)

    c = np.dot(a,b)
    print("Dot product:", c)

def e_14():
    A = np.matrix(np.random.randint(0,10,(4,3)))
    y = np.random.randint(0,10,3)
    print(A)
    print(y)

    R = np.hstack((A, np.tile(y,(A.shape[0],1))))
    print(R)

    # print(A + y)


if __name__ == "__main__":
    e_14()