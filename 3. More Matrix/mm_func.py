def transpose(A):
    """
    행렬의 전치 행렬
    input:  전치 행렬할 행렬 A
    output: 전치 행렬 At
    """
    n = len(A)      # row
    p = len(A[0])   # column

    At = []
    for i in range(0, p):
        row = []
        for j in range(0, n):
            val = A[j][i]
            row.append(val)
        At.append(row)

    return At


def diag(A):
    """
    행렬의 대각행렬
    input: 대각행렬을 구하고자하는 행렬 A
    output: 행렬 A의 대각 행렬 D
    """
    n = len(A)
    D = []

    for i in range(0, n):
        row = []
        for j in range(0, n):
            if i == j:
                row.append(A[i][j])
            else:
                row.append(0)
        D.append(row)
    return D

def diag_ele(A):
    """
    대각 원소 구하기
    input: 대각 원소를 구하는 행렬 A
    output: 행렬 A의 대각 원소 리스트 d
    """
    n = len(A)
    d = []
    for i in range(0, n):
        d.append(A[i][i])
    return d


def ele2diag(a):
    """
    대각 원소로 대각 행렬 변환
    input: 대각 원소 리스트 a
    output: n x n 크기의 대각 행렬 D
    """
    n = len(a)
    D = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            if i == j:
                row.append(a[i])
            else:
                row.append(0)
        D.append(row)
    return D


def identity(n):
    """
    단위 행렬 생성
    input: 단위 행렬의 크기 n
    output: n x n 크기의 단위 행렬 I
    """
    I = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        I.append(row)
    return I


def zero_mat(n, p):
    """
    영행렬 생성
    input: 생성하고자 할 영 행렬의 크기 n행, p열
    output: n x p 크기의 영행렬 Z
    """
    Z = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            row.append(0)
        Z.append(row)
    return Z


def u_tri(A):
    """
    상 삼각행렬 변환
    input: 상 삼각 행렬로 변환하고자하는 행렬 A
    output: 행렬 A를 상 삼각 행렬로 변환시킨 행렬 utri
    """
    n = len(A)
    p = len(A[0])
    utri = []

    for i in range(0, n):
        row = []
        for j in range(0, p):
            if i>j:
                row.append(0)
            else:
                row.append(A[i][j])
        utri.append(row)
    return utri


def l_tri(A):
    """
    하 삼각행렬 변환
    input: 하 삼각 행렬로 변환하고자하는 행렬 A
    output: 행렬 A를 하 삼각 행렬로 변환시킨 행렬 utri
    """
    n = len(A)
    p = len(A[0])
    ltri = []

    for i in range(0, n):
        row = []
        for j in range(0, p):
            if i<j:
                row.append(0)
            else:
                row.append(A[i][j])
        ltri.append(row)
    return ltri


def toeplitz(a, b):
    """
    토플리츠 행렬 변환
    input: 토플리츠 행렬로 변환하고자 하는 리스트 a, b
    output: 리스트 a, b를 이용해 만든 토플리츠 행렬 A
    """
    n1 = len(a)
    n2 = len(b)
    A = []
    for i in range(0, n1):
        row = []
        for j in range(0, n2):
            if i > j:
                row.append(a[i-j])
            else:
                row.append(b[j-i])
        A.append(row)
    return A


def u_bidiag(A):
    """
    upper bidiagonal 행렬
    input: 행렬 A
    output: 행렬 A의 upper bidiagonal 행렬 res
    """
    n = len(A)
    p = len(A[0])

    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            if i > j or j - i > 1:
                row.append(0)
            else:
                row.append(A[i][j])
        res.append(row)
    return res


def l_bidiag(A):
    """
    lower bidiagonal 행렬
    input: 행렬 A
    output: 행렬 A의 lower bidiagonal 행렬 res
    """
    n = len(A)
    p = len(A[0])

    res = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            if i < j or i - j > 1:
                row.append(0)
            else:
                row.append(A[i][j])
        res.append(row)

    return res




# 하우스 홀더 함수
def inner_product(a, b):
    """
    벡터의 내적
    input: 내적할 벡터 리스트 a, b
    output: 내적 결과 res
    """
    n = len(a)
    res = 0
    for i in range(0, n):
        res += a[i]*b[i]
    return res


def outer_product(a, b):
    """
    벡터의 외적
    input: 외적할 벡터 리스트 a, b
    output: 외적 결과 res
    """
    res = []
    n1 = len(a)
    n2 = len(b)
    for i in range(0, n1):
        row = []
        for j in range(0, n2):
            val = a[i]*b[j]
            row.append(val)
        res.append(row)
    return res


def subtract(A, B):
    """
    행렬의 뺄셈
    input: 행렬의 뺄셈을 수행할 행렬 A, B
    output: 행렬 A와 행렬 B의 뺄셈 결과인 res
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


def householder(v):
    """
    하우스홀더 행렬
    input: 하우스홀더 행렬을 생성할 리스트 v
    output: 하우스홀더 행렬 H
    """
    n = len(v)
    outer_mat = outer_product(v, v)
    inner_val = inner_product(v, v)
    V = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            val = (2/inner_val) * outer_mat[i][j]
            row.append(val)
        V.append(row)
    H = subtract(identity(n), V)
    return H


