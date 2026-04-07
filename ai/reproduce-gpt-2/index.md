# 拉格朗日插值


## 1
可以把这两个层想象成：

self.c_attn: "问题生成器"
把输入信息转换成：要查询什么(Q)、用什么匹配(K)、提取什么信息(V)
self.c_proj: "答案整合器"
把各个"专家"(注意力头)的答案整合成最终结果
```python
B:batch_size
T:Sequence Length
C:Embedding Dimension
attn = nn.Linear(n_embd, 3*n_embd)
qkv = self.c_attn(x)
q, k, v = qkv.spilt(self.n_embd, dim = 2)
c_proj = nn.Linear(n_embd, n_embd)

```


