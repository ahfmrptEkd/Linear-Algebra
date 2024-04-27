# 행렬 덧셈 함수
def add(A,B) -> list():
    """
    행렬의 덧셈
    input: 행렬 덧셈을 수행할 행렬 A, B 
    output: 덧셈 결과인 행렬 A
    """ 
    n = len(A)
    p = len(A[0])

    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = A[i][j] + B[i][j]
            row.append(val)

        res.append(row)
    return res


# 행렬 뺄셈 함수
def subtract(A, B) -> list():
    """
    행렬의 뺄셈
    input: 행렬의 뺄셈 수행할 행렬 A, B
    output: 뺄셈한 행렬 C
    """
    n = len(A)
    p = len(A[0])

    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = A[i][j] - B[i][j]
            row.append(val)

        res.append(row)

    return res


# 행렬의 스칼라 곱
def scalar_mul(b:int, A:list()) ->list():
    """
    행렬의 스칼라 곱
    input: 스칼라 곱을 수행할 스칼라 b, 행렬 A
    output: 곱셈의 결과 행렬 C
    """
    n = len(A)
    p = len(A[0])

    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = b * A[i][j]
            row.append(val)

        res.append(row)

    return res


# 행렬의 원소 곱
def ele_product(A:list, B:list)-> list():
    """
    행렬의 원소 곱
    input: 행렬의 원소 곱을 수행할 행렬 A, B
    output: 원소곱 결과 행렬 C
    """
    n = len(A)
    p = len(A[0])

    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            val = A[i][j] * B[i][j]
            row.append(val)
        res.append(row)
    return res


# 행렬 곱
def matmul(A, B)-> list():
    """
    행렬과 행렬의 곱
    input: 행렬곱 할 행렬 A, B
    output: 행렬곱 결과 C
    """
    n = len(A)
    p1 = len(A[0])
    p2 = len(B[0])

    res = []

    for i in range(0, n):
        row = []
        for j in range(0, p2):
            val = 0
            for k in range(0, p1):
                val += A[i][k] * B[k][j]
            row.append(val)
        res.append(row)
    return res