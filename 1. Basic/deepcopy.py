def deepcopy(a: list()) -> list():
    """
    ���� ���� ����
    input : deepcopy�� �ϰ��� �ϴ� matrix list a
    output : copied ��� matrix list res
    """
    if type(a[0]) == list:  # a �� ����̶��?
        n = len(a)  # ��
        p = len(a[0])   # ��
        res = zero_mat(n, p)
        for i in range(0, n):
            for j in range(0, p):
                res[i][j] = a[i][j]
        return res
    
    else:                   # a �� ���Ͷ��?
        n = len(a)
        res = []
        for i in range(0, n):
            res.append(a[i])
        return res


def zero_mat(n, p):
    """
    �� ��� ����
    input : ������ ����� �� & �� ũ��
    output : (n * p) ũ���� �� ���
    """
    z = []
    for i in range(0, n):
        row = []
        for j in range(0, p):
            row.append(0)
        z.append(row)
    return z