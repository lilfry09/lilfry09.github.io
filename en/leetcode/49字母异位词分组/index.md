# Letter anagram grouping

## Question description
You are given an array of strings, and you are asked to combine the anagrams together. The list of results can be returned in any order.

 

Example 1:

Input: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]

Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

explain:

There are no strings in strs that can be rearranged to form "bat".
The strings "nat" and "tan" are anagrams because they can be rearranged to form each other.
The strings "ate" , "eat" and "tea" are anagrams because they can be rearranged to form each other.
Example 2:

Input: strs = [""]

Output: [[""]]

Example 3:

Input: strs = ["a"]

Output: [["a"]]

## Problem-solving ideas
1. **Key Points**
   - Idea 1: Hash table, use the sorted result of each string as key to group anagrams
   - Idea 2: Counting, the number of occurrences of each character in each string is used as the key, and the anagrams are grouped

## Code implementation (c++)
```c++
class Solution {
public:
    std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string>& strs) {
        // Create a hash map: key = sorted string, value = list of anagrams
        std::unordered_map<std::string, std::vector<std::string>> map;

        // Iterate over every input string
        for (const std::string& str : strs) {
            // Create a copy for sorting to generate the key
            std::string key = str;
            std::sort(key.begin(), key.end());

            // Add the original string `str` to the group mapped by `key`
            // If the key does not exist, C++ map creates an empty vector automatically
            map[key].push_back(str);
        }

        // Create the output list
        std::vector<std::vector<std::string>> result;
        
        // Traverse the hash map and append each group (map value) to the result
        // Structured bindings in C++17 make the code cleaner
        for (auto const& [key, val] : map) {
            result.push_back(val);
        }

        return result;
    }
};
```

