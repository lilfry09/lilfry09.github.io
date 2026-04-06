# 拉格朗日插值


## 拉格朗日插值法简介

已知函数y=f(x)在n+1个不同点(x0,y0),(x1,y1),...,(xn,yn)(x0 != x1 != ... != xn),处的函数值y0,y1,...yn,求次数不大于n的多项式P(x)满足P(xi)=yi(i=0,1,...,n).

#### 唯一性
Cramer法则，Vandermonde行列式
D=
| 1 | x0 | x0^2 | ... | x0^n |
| 1 | x1 | x1^2 | ... | x1^n |
| 1 | x2 | x2^2 | ... | x2^n |  != 0
| ... | ... | ... | ... | ... |
| 1 | xn | xn^2 | ... | xn^n |

#### 拉格朗日插值多项式
P(x)=y0L0(x)+y1L1(x)+...+ynLn(x)

其中Li(x) = ∏(x - xj)/(xi - xj), j=0,j!=i to n
Li(xi) = 1
Li(xj) = 0, i!=j

```python
from scipy.interpolate import lagrange
import numpy as np

Ln = lagrange([1, 2], [3, 4])
print(Ln)
```

![拉格朗日插值法示例](/images/拉格朗日插值.png)

