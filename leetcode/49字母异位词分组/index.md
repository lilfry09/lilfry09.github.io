# 字母异位词分组

## 题目描述
给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

 

示例 1:

输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]

输出: [["bat"],["nat","tan"],["ate","eat","tea"]]

解释：

在 strs 中没有字符串可以通过重新排列来形成 "bat"。
字符串 "nat" 和 "tan" 是字母异位词，因为它们可以重新排列以形成彼此。
字符串 "ate" ，"eat" 和 "tea" 是字母异位词，因为它们可以重新排列以形成彼此。
示例 2:

输入: strs = [""]

输出: [[""]]

示例 3:

输入: strs = ["a"]

输出: [["a"]]

## 解题思路
1. **关键点**  
   - 思路1：哈希表，使用每个字符串排序后的结果作为key，将异位词分组
   - 思路2：计数，每个字符串的每个字符出现的次数作为key，将异位词分组

## 代码实现（c++）
```c++
class Solution {
public:
    std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string>& strs) {
        // 创建一个哈希表，键是排序后的字符串，值是字母异位词列表
        std::unordered_map<std::string, std::vector<std::string>> map;

        // 遍历输入的每一个字符串
        for (const std::string& str : strs) {
            // 创建一个副本用于排序，得到 key
            std::string key = str;
            std::sort(key.begin(), key.end());

            // 将原始字符串 str 添加到 key 对应的分组中
            // 如果 key 不存在，C++ 的 map 会自动创建一个空 vector
            map[key].push_back(str);
        }

        // 创建结果列表
        std::vector<std::vector<std::string>> result;
        
        // 遍历哈希表，将每个分组（map中的值）添加到结果列表中
        // C++17 的结构化绑定，让代码更简洁
        for (auto const& [key, val] : map) {
            result.push_back(val);
        }

        return result;
    }
};
```

