# Useful Python Tips


一些有用的Python小技巧，以及学习过程中遇到的问题和记录。

```python
numbers = list(range(1,11))
print(numbers)
squared = [x**2 for x in numbers] # 列表解析
print(squared\n)

a = []
for x in numbers:
    s = x**2
    a.append(s)
print(a)
```

```python
favorite_languages = {
    'A':'Python',
    'B':'C++',
    'C':'Java',
    'D':'JavaScript',
    'E':'Ruby',
}
    for name, language in favorite_languages.items():
        print(name.title() + "'s favorite language is " + language.title() + ".")
```
