# Find duplicate subtrees


## Question description
Given the root node root of a binary tree, return all duplicate subtrees.

For duplicate subtrees of the same type, you only need to return the root node of any one of them.

If two trees have the same structure and the same node values, they are considered duplicates.

https://assets.leetcode.com/uploads/2020/08/16/e1.jpg

Input: root = [1,2,3,4,null,2,4,null,null,4]
Output: [[2,4],[4]]

## Problem-solving ideas

## Use postorder traversal + hashing!

Idea:
1. Give each subtree an "ID card" (a string traversed in post-order)
2. Record the number of occurrences of each subtree (hash, counter)

### **Why does this method work? **

1. **String description uniquely determines the subtree structure**
For example, `"2,4,#,#,#"` can only correspond to one binary tree (root->left->right)
    
2. **Counter accurately captures repetitions**
When the same string appears a second time, it means that the subtree with exactly the same structure appears twice.

### **1. Why do you think of using strings to represent subtrees? **

- **Intuitive requirements**: To determine whether two subtrees are the same, you need to compare their **complete structures** (including node values ​​and left and right subtrees).
- **Characteristics of strings**:
A string can **uniquely encode the structure of a tree** (for example `"2,#,#"` can only correspond to a subtree of single node 2).
- **Natural adaptation of recursion**:
Tree traversal itself is recursive, and string concatenation is naturally suitable for recursion (left + right + root = complete structure).

---

### **2. Why use postorder traversal? **

- **Postorder (left→right→root) order**:
Only by first knowing the serialization results of the left and right subtrees can the string of the current subtree be spliced.
(For example, you must first know `4,#,#` and `#` before you can spell `2,4,#,#,#`)

---

### **3. Why use a hash table (or counter)? **

- **Quick judgment**:
When the serialized string appears for the second time, it means that a subtree with the same structure has been encountered.
(The hash table can determine whether it is repeated in O(1) time, which is much more efficient than brute force traversal)

---

### **4. Why should empty nodes be represented by `#`? **

- **Avoid ambiguity**:
For example, the left side of the subtree `2` is empty, the right side is `3`, and the serialization is `"2,#,3,#,#"`;
If `#` is not used, `"2,3"` may be mistaken for `2` and the left side is `3`.

---

### **5. Source of inspiration for this method**

- **"Serialization" in the database**:
Similar to converting complex data into string storage (such as JSON).
- **"Syntax tree" in the compiler**:
When the code is compiled, string hashing is used to optimize the detection of repeated structures in the syntax tree.
- **Natural human thinking**:
Imagine you were trying to describe the structure of a tree to someone. What would you say?
→ "The root is 2, the left is 4, and the right is empty" → Corresponds exactly to `"2,4,#,#,#"`!


## Code implementation
```c++
class Solution {

public:

    vector<string> all_subtrees;  // Store serialized strings of all subtrees

    unordered_map<string, int> count;  // Record how many times each serialization appears

    vector<TreeNode*> result;     // Store root nodes of duplicate subtrees

  

    string serialize(TreeNode* node) {

        if (node == nullptr) return "#";

        string left = serialize(node->left);

        string right = serialize(node->right);

        // Serialized representation of the current subtree

        string subtree = to_string(node->val) + "," + left + "," + right;

        // Check whether it has appeared before

        if (count[subtree] == 1) {

            result.push_back(node);

        }

        count[subtree]++;

        return subtree;

    }

  

    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {

        serialize(root);

        return result;

    }

};
```

