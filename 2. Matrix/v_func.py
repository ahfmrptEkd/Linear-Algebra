# ?? ?? ??
def v_add(u, v):
    """
    ??? ??
    input: ???? ?? ?? u, v
    output: ?? ?? ?? w
    """
    n = len(u)
    w = []

    for i in range(0, n):
        val = u[i] + v[i]
        w.append(val)
    
    return w



# ?? ?? ??
def v_subtract(u, v):
    """
    ??? ??
    input: ??? ?? ?? u, v
    output: ?? ?? ?? w
    """
    n = len(u)
    w = []

    for i in range(0, n):
        val = u[i] - v[i]
        w.append(val)
    
    return w



# ?? ??
def scalar_v_mul(a, u):
    """
    ??? ??? ?
    input: scalar a, vector list u
    output: ? ?? w
    """
    n = len(u)
    w = []

    for i in range(0, n):
        val = a * u[i]
        w.append(val)
    
    return w


# ?? ?? ? ??
def v_mul(u, v):
    """
    ??? ?? ?
    input: ??? ? ?? u, v
    output: ?? ?? w
    """
    n = len(u)
    w = []

    for i in range(0, n):
        val = u[i] * v[i]
        w.append(val)
    
    return w


# ??? ??
def v_div(u, v):
    """
    ?? element-wise ???
    input: ???? ?? ?? u, v
    output: ?? ??? ?? w
    """
    n = len(u)
    w = []

    for i in range(0, n):
        val = u[i] / v[i]
        w.append(val)
    
    return w