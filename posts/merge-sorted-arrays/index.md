# 合并两个有序数组 - LeetCode 88


## 题目描述

**难度**：简单  
**题号**：88  
**链接**：[合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/description/)

## 解题思路

使用尾指针从后往前遍历，比较两个数组的元素，将较大的元素放到 nums1 的末尾，然后移动指针，重复这个过程，直到遍历完 nums2。

如果 nums2 还有剩余元素，继续将它们放到 nums1 的末尾。

## 代码实现

```cpp
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int k = m + n - 1;
        
        while (i >= 0 && j >= 0) {
            if (nums1[i] > nums2[j]) {
                nums1[k--] = nums1[i--];
            } else {
                nums1[k--] = nums2[j--];
            }
        }
        
        while (j >= 0) {
            nums1[k--] = nums2[j--];
        }
    }
};
```

## 复杂度分析

- **时间复杂度**：O(m + n)
- **空间复杂度**：O(1)

