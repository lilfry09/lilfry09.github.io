# 寻找数组最大值和最小值 


## 题目描述
伪代码：
FindMinMax(A, left, right)
    if left = right
        return A[left], A[left]  # 只有一个元素，它既是最大值也是最小值
    
    mid = (left + right) / 2
    left_min, left_max = FindMinMax(A, left, mid)      # 递归处理左半部分
    right_min, right_max = FindMinMax(A, mid + 1, right) # 递归处理右半部分
    
    total_min = min(left_min, right_min)  # 比较左右部分的最小值
    total_max = max(left_max, right_max)  # 比较左右部分的最大值
    
    return total_min, total_max
