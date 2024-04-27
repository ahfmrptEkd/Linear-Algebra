# 복사

<br>

## 얕은 복사
mutable 한 객체를 복사하여 복사한 객체의 변경이 서로에게 영향을 주는 문제가 있다.  

이러한 문제를 방지하기 위해 copy를 이용할 수 있다.

```python
a = [1,2,3,4,5]
b = a[:]    # a 객체를 복사하는 hard code

>>>print(a)
[1, 2, 3, 4, 5]

>>>print(b)
[1, 2, 3, 4, 5]
```

```python
>>>print(id(a))
1299127672896

>>>print(id(b))
1299127701312


>>>print(id(a[0]))
1299074777328

>>>print(id(b[0]))
1299074777328
```
위 처럼 확인해보면 서로 다른 메모리에 저장되어 있음  

하지만 객체의 element를 확인해보면 그렇지 않음을 알 수 있음


```python
# 파이썬의 제공하는 copy 라이브러리를 통해 얕은 복사를 실행가능
import copy

a = [1,2,3,4,5]
b = copy.copy(a)

>>>print(id(a))
1299127699264
>>>print(id(b))
1299127684544

b[3] = 8

>>>print(b)
[1, 2, 3, 8, 5]
>>>print(a)
[1, 2, 3, 4, 5]


>>>print(id(b[3]))
1299074777552
>>>print(id(a[3]))
1299074777424
```
이 처럼 얕은 복사를 한경우는 서로 객체에서 변경점이 생긴다면; 영향을 끼치지 않고 새로운 메모리에 할당이 되는것을 확인할 수 있다.

<br>

### 문제점
하지만 얕은 복사도 문제가 없는 것은 아니다.  

얕은 복사는 mutable 한 객체의 immutable한 객체를 구성하는 경우에는 **유용** 하다.  

반대로 mutable 한 객체의 또 다른 mutable한 객체를 구성하는 경우는 문제가 있다.

```python
c = [[1, 3], [2, 4]]
d = copy.copy(c) # 얕은 복사

# 서로 다른 메모리에 할당이 된 것이 보임. 하지만 그 안은?
>>>print(id(c))
1299149872576
>>>print(id(d))
1299149638848


>>>print(id(c[0][0]))
1299074777328
>>>print(id(d[0][0]))
1299074777328



d[0][0] = 7
>>>print(d)
[[7, 3], [2, 4]]
>>>print(c)
[[7, 3], [2, 4]]


# 같은 원소 위치더라도, 주소가 바뀐 것을 확인 할 수 있다.
>>>print(id(c[0][0]))
1299074777520
>>>print(id(d[0][0]))
1299074777520
```

이 처럼 안에 있는 또 다른 mutable 객체에 대해 같은 주소를 공유하는 것을 볼 수 있다.  

이 말은 d의 원소가 바뀌면, c의 원소 또한 바뀌는 것을 의미한다.  

이것을 방지하기 위해서는 깊은 복사 **deepcopy**를 이용해야 한다.

<br><br><br>

## 깊은 복사
```python
import copy

e = [[1, 2], [3, 4]]
f = copy.deepcopy(e)
>>>print(e)
[[1, 2], [3, 4]]
>>>print(f)
[[1, 2], [3, 4]]


>>>print(id(e))    # 첫 주소는 얕은 복사처럼 같은 주소를 지니지만,
1299149620288
>>>print(id(f))
1299126378816


>>>print(id(e[0][0]))  # 얕은 복사와는 달리 안에 있는 객체들 마저도 다른 주소를 가지고 있어 공유하지 않음
1299074777328
>>>print(id(f[0][0]))
1299074777328


f[0][0] = 5
>>>print(f)
[[5, 2], [3, 4]]
>>>print(e)
[[1, 2], [3, 4]]

>>>print(id(f[0][0]))
1299074777328
>>>print(id(e[0][0]))
1299074777456
```

### 깊은 복사 함수 implement

```python
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


>>>A = [[1,2,3], [4,5,6], [7,8,9]] # Matrix
>>>B = deepcopy(A)

>>>print(B)
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]


>>>a = [1,2,3]
>>>b = deepcopy(a)

>>>print(b)
[1, 2, 3]
```