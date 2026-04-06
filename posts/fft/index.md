# 快速傅里叶变换（FFT）详解


## 1. 什么是傅里叶变换？

傅里叶变换是一种将信号从时域转换到频域的数学工具。它可以将任意周期信号分解为一系列正弦波和余弦波的叠加。

### 离散傅里叶变换（DFT）

对于长度为 $N$ 的序列 $x[0], x[1], ..., x[N-1]$，其 DFT 定义为：

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i \frac{2\pi}{N} kn}, \quad k = 0, 1, ..., N-1$$

其中 $i$ 是虚数单位，$e^{-i\theta} = \cos\theta - i\sin\theta$（欧拉公式）。

### 时间复杂度问题

直接计算 DFT 需要 $O(N^2)$ 的时间复杂度，因为：
- 需要计算 $N$ 个频率分量 $X[k]$
- 每个 $X[k]$ 需要对 $N$ 个时域样本求和

当 $N$ 很大时（如音频处理中 $N=4096$），计算量非常大。

---

## 2. 快速傅里叶变换（FFT）

FFT 是一种高效计算 DFT 的算法，由 Cooley 和 Tukey 在 1965 年提出。它将复杂度从 $O(N^2)$ 降低到 $O(N\log N)$。

### 核心思想：分治法

FFT 的关键在于利用**对称性**和**周期性**将大问题分解为小问题。

#### 假设 $N = 2^m$（$N$ 为 2 的幂次）

将序列按奇偶分组：
- 偶数项：$x[0], x[2], x[4], ..., x[N-2]$
- 奇数项：$x[1], x[3], x[5], ..., x[N-1]$

则 DFT 可以表示为：

$$X[k] = \sum_{n=0}^{N/2-1} x[2n] \cdot e^{-i \frac{2\pi}{N} k(2n)} + \sum_{n=0}^{N/2-1} x[2n+1] \cdot e^{-i \frac{2\pi}{N} k(2n+1)}$$

化简后：

$$X[k] = E[k] + e^{-i \frac{2\pi}{N} k} \cdot O[k]$$

其中：
- $E[k]$：偶数项的 $N/2$ 点 DFT
- $O[k]$：奇数项的 $N/2$ 点 DFT
- $e^{-i \frac{2\pi}{N} k}$ 称为**旋转因子**（twiddle factor）

### 递归结构

1. 原问题：$N$ 点 DFT
2. 分解为：2 个 $N/2$ 点 DFT
3. 继续分解，直到只有 1 个点（$T(1) = O(1)$）

**递推式**：

$$T(N) = 2T(N/2) + O(N)$$

根据主定理，$T(N) = O(N\log N)$。

---

## 3. FFT 算法实现

### 蝴蝶操作（Butterfly Operation）

FFT 的基本计算单元称为蝴蝶操作：

给定两个复数 $a$ 和 $b$，以及旋转因子 $W = e^{-i \frac{2\pi}{N} k}$：

$$
\begin{cases}
a' = a + W \cdot b \\
b' = a - W \cdot b
\end{cases}
$$

这个操作形状像蝴蝶的翅膀，因此得名。

### Python 实现（递归版本）

```python
import numpy as np

def fft_recursive(x):
    """递归实现的 FFT
    Args:
        x: 输入序列（长度必须是 2 的幂次）
    Returns:
        频域表示
    """
    N = len(x)
    
    # 递归基：长度为 1
    if N == 1:
        return x
    
    # 确保 N 是 2 的幂次
    if N % 2 != 0:
        raise ValueError("输入长度必须是 2 的幂次")
    
    # 分治：奇偶分组
    even = fft_recursive(x[0::2])  # 偶数索引
    odd = fft_recursive(x[1::2])   # 奇数索引
    
    # 合并：计算旋转因子
    T = np.exp(-2j * np.pi * np.arange(N) / N)
    
    # 蝴蝶操作
    return np.concatenate([
        even + T[:N//2] * odd,
        even + T[N//2:] * odd
    ])

# 测试
if __name__ == "__main__":
    # 生成测试信号
    N = 8
    t = np.arange(N)
    signal = np.sin(2 * np.pi * 1 * t / N) + 0.5 * np.sin(2 * np.pi * 2 * t / N)
    
    # FFT
    freq_domain = fft_recursive(signal)
    
    # 验证结果（与 NumPy 对比）
    np_fft = np.fft.fft(signal)
    print("自定义 FFT:", np.round(freq_domain, 4))
    print("NumPy FFT: ", np.round(np_fft, 4))
    print("误差:", np.max(np.abs(freq_domain - np_fft)))
```

### 迭代版本（Cooley-Tukey 算法）

```python
def fft_iterative(x):
    """迭代实现的 FFT（更高效）"""
    N = len(x)
    
    # 位反转排列（bit-reversal permutation）
    def bit_reverse(n, bits):
        return int(bin(n)[2:].zfill(bits)[::-1], 2)
    
    bits = int(np.log2(N))
    x = np.array([x[bit_reverse(i, bits)] for i in range(N)])
    
    # 迭代计算
    for s in range(1, bits + 1):
        m = 2 ** s
        Wm = np.exp(-2j * np.pi / m)
        for k in range(0, N, m):
            W = 1
            for j in range(m // 2):
                t = W * x[k + j + m // 2]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + m // 2] = u - t
                W *= Wm
    
    return x
```

---

## 4. FFT 的应用

### 4.1 信号处理
- **音频压缩**：MP3、AAC 等格式使用 FFT 分析频率成分
- **频谱分析**：查看信号的频率组成

### 4.2 图像处理
- **JPEG 压缩**：使用离散余弦变换（DCT，FFT 的变种）
- **图像滤波**：频域滤波比时域卷积更高效

### 4.3 多项式乘法
两个多项式相乘的朴素算法是 $O(N^2)$，使用 FFT 可以降低到 $O(N\log N)$：

1. 将多项式系数视为时域信号
2. FFT 转换到频域
3. 频域中逐点相乘
4. 逆 FFT（IFFT）转换回时域

```python
def polynomial_multiply(a, b):
    """使用 FFT 实现多项式乘法"""
    # 补零到 2 的幂次
    n = 1
    while n < len(a) + len(b):
        n *= 2
    
    a_padded = np.pad(a, (0, n - len(a)))
    b_padded = np.pad(b, (0, n - len(b)))
    
    # FFT
    A = np.fft.fft(a_padded)
    B = np.fft.fft(b_padded)
    
    # 频域相乘
    C = A * B
    
    # IFFT
    c = np.fft.ifft(C).real
    
    return c[:len(a) + len(b) - 1]

# 示例：(x + 2)(x + 3) = x^2 + 5x + 6
a = [2, 1]  # 2 + x
b = [3, 1]  # 3 + x
result = polynomial_multiply(a, b)
print("结果:", result)  # [6, 5, 1] 对应 6 + 5x + x^2
```

---

## 5. 复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 朴素 DFT | $O(N^2)$ | $O(N)$ |
| FFT | $O(N\log N)$ | $O(N\log N)$（递归）/ $O(N)$（迭代） |

### 实际性能对比

```python
import time

def benchmark():
    sizes = [256, 512, 1024, 2048, 4096]
    
    for N in sizes:
        x = np.random.rand(N)
        
        # FFT
        start = time.time()
        np.fft.fft(x)
        fft_time = time.time() - start
        
        print(f"N={N}: FFT 用时 {fft_time*1000:.2f} ms")

benchmark()
```

---

## 6. 注意事项

### 6.1 输入长度
- FFT 要求输入长度为 $2^m$
- 不足时需要**补零**（zero-padding）

### 6.2 频谱泄漏
- 对非周期信号使用 FFT 会产生频谱泄漏
- 解决方法：加**窗函数**（如汉宁窗、汉明窗）

```python
# 加窗示例
window = np.hanning(N)
signal_windowed = signal * window
fft_result = np.fft.fft(signal_windowed)
```

### 6.3 采样定理
- 采样频率 $f_s$ 必须 $\geq 2f_{max}$（奈奎斯特定理）
- 否则会产生**混叠**（aliasing）

---

## 7. 总结

**FFT 的优势**：
- ✅ 将 $O(N^2)$ 降低到 $O(N\log N)$
- ✅ 广泛应用于信号处理、图像处理、科学计算
- ✅ 是现代数字信号处理的基石



