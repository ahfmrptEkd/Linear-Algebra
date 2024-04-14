def deepcopy(a: list()) -> list():
    """
    깊은 복사 구현
    input : deepcopy를 하고자 하는 matrix list a
    output : copied 결과 matrix list res
    """
    if type(a[0]) == list:  # a 가 행렬이라면?
        n = len(a)  # 행
        p = len(a[0])   # 열
        res = zero_mat(n, p)
        for i in range(0, n):
            for j in range(0, p):
                res[i][j] = a[i][j]
        return res
    
    else:                   # a 가 벡터라면?
        n = len(a)
        res = []
        for i in range(0, n):
            res.append(a[i])
        return res


def zero_mat(n, p):
    """
    영 행렬 생성
    input : 생성할 행렬의 행 & 열 크기
    output : (n * p) 크기의 영 행렬
    """
    z = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            row.append(0)
        z.append(row)
    return z