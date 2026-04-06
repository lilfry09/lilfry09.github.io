# 如何用hugo构建自己的博客

先在GitHub上创建两个仓库，一个private，一个public，public的仓库记得把名字命名为`yourusername.github.io`（新手请一定这么做）
假设你的私有仓库名为 `blogsource`，公开仓库是 `yourusername.github.io`，以下是部署全流程
（过程中有不明白的地方请询问deepseek）
## 准备工作

1️⃣下载hugo和git（安装流程请询问deepseek）
2️⃣在D盘创建一个文件夹，用于放置你的本地文件，在powershell里面cd到这个文件夹，然后我们就可以开始拉！

## 注意全过程不要在github网页上更改文件！！使用vscode等文本编辑器本地推送上去


### **🚀 完整流程（适配你的仓库名）**

#### **1️ 初始化 Hugo 项目（私有仓库 `blogsource`）**
在powershell或者vscode终端运行一下代码，记得是在你cd到的这个文件夹运行
例如：我的是在D:\MyBlog下，然后输入以下代码
```bash
# 创建项目并初始化 Git
hugo new site blogsorce
cd blogsorce
git init（初始化）
git remote add origin https://github.com/lilfry09/blogsource.git(把这里的lilfry09换成你的用户名)
```

#### **2️ 添加主题（以 LoveIt 为例）**
这里的模板可以自己任意选择
```bash
git submodule add https://github.com/dillonzq/LoveIt themes/LoveIt
```

#### **3️ 配置 `hugo.toml`**
```toml
baseURL = "https://lilfry09.github.io/"  # 必须带斜杠！换成你的github用户名
title = "My Blog"# 网站标题
theme = "LoveIt"
publishDir = "public"  # 静态文件输出目录
```
### **4 创建测试文章**


`hugo new posts/hello-world.md`

### 编辑 `content/posts/hello-world.md`，添加内容：
---
title: "Hello World"
date: 2024-01-01
draft: false   `#记得draft要设成false才能显示`
---
## **5 本地测试**

`hugo server -D  # 启动本地预览`

访问 `http://localhost:1313`，确认博客正常显示。

## **6 推送源码到私有仓库 `blogsource`**

### **(1) 关联 GitHub 私有仓库**

📌 如果之前关联过错误的远程仓库，先删除：

`git remote remove origin`

再使用：
`git remote add origin https://github.com/lilfry09/blogsource.git`

### **(2) 提交代码**

`git add .` 
`git commit -m "Initial commit: Hugo project with LoveIt theme"`


`#强制推送（适用于全新仓库）`
`git push -u origin main --force`


或（如果分支已存在）：

`# 普通推送` 
`git push -u origin main`

✅ 现在源码已安全存放在私有仓库 `blogsource`。

### **🔍 验证是否成功**

1. **检查 GitHub 仓库**  
    访问 `https://github.com/lilfry09/blogsource`，确认文件已出现。
    
2. **查看分支状态**  
    运行：
    `git branch -a`
    
    应显示：
    `* main   remotes/origin/main`

#### 7 **生成静态文件并推送到公开仓库**
```bash
hugo -D  # 生成 public/
cd public #cd到刚生成的这个文件夹
git init
git add .
git commit -m "Deploy to GitHub Pages"
git remote add origin https://github.com/lilfry09/lilfry09.github.io.git 
git branch -M main # 确保分支名是 main
# 用你自己的用户名
git push -u origin main --force  # 强制覆盖（首次部署）
```

#### **设置 GitHub Pages**
1. 进入 `lilfry09.github.io` 仓库 → Settings → Pages  
   - **Branch**: `main`，根目录 `/`  
2. 等待 1-2 分钟，访问：  
   🔗 **https://lilfry09.github.io**
（改成你的用户名）

---

### **⚡ 自动化部署（推荐）**（这步先不要做，上面那个先成功再说）
在私有仓库 `blogsource` 中添加 GitHub Actions：  
## Step1：ssh生成密钥对
在git bash中运行
`ssh-keygen -t ed25519 -C "actions-deploy-key" -f ~/.ssh/gh-pages-key`

####  **获取生成的密钥**：

- **私钥**（用于 GitHub Secrets）：
    
    `cat ~/.ssh/gh-pages-key`
    
- **公钥**（用于 Deploy Keys）：
    
    `cat ~/.ssh/gh-pages-key.pub`
### **📥 将密钥配置到 GitHub**

#### 1. **添加公钥到目标仓库**（`lilfry09.github.io`）：

- 进入仓库 → `Settings` → `Deploy Keys` → `Add deploy key`
- **Title**: `ACTIONS_DEPLOY_KEY`
- **Key**: 粘贴 `gh-pages-key.pub` 的内容
- **勾选** `Allow write access`（必须勾选！）

#### 2. **添加私钥到源仓库**（`blogsource`）：

- 进入仓库 → `Settings` → `Secrets and variables` → `Actions` → `New repository secret`
- **Name**: `ACTIONS_DEPLOY_KEY`
- **Value**: 粘贴 `gh-pages-key` 的内容（包含 `-----BEGIN OPENSSH PRIVATE KEY-----` 等完整内容）

#### **检查本地仓库分支**

`确认本地分支是否为 main git branch 
`如果不是，切换到 main 分支 git checkout -b main`

在本地用vscode（或者其他文本编辑器）在你的D:\MyBlog\blogsource创建文件 `.github/workflows/deploy.yml`：
```
name: Deploy Hugo Site to GitHub Pages

on:
  push:
    branches: [ "main" ]  

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: true  # 如果主题是子模块必选
          fetch-depth: 0    # 获取完整历史

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v2
        with:
          hugo-version: '0.145.0'  # 改为你的 Hugo 版本

      - name: Build
        run: hugo --minify  # 生成静态文件到 public/

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          deploy_key: ${{ secrets.ACTIONS_DEPLOY_KEY }}  # 对应你之前存的私钥名
          external_repository: lilfry09/lilfry09.github.io  # 目标公开仓库
          publish_dir: ./public  # Hugo 输出目录
          keep_files: false      # 完全覆盖目标仓库
```

### 验证工作流是否生效

1. **提交并推送文件**：
    `git add .github/workflows/deploy.yml` 
    `git commit -m "Add GitHub Actions workflow"`
    `git push origin main  # 或你的分支名`
    
2. **在 GitHub 上检查**：
    - 进入 `blogsource` 仓库 → `Actions` 标签页
    - 应该会出现 **“Deploy Hugo Site to GitHub Pages”** 工作流
    - 点击最新运行记录，查看是否有错误（绿色✅表示成功）



推送代码触发自动化：
```bash
git add . 
git commit -m "Add GitHub Actions"
git push origin main
```

然后再创建新blog的时候就不用手动生成public再推送了，在本地content文件夹内添加md.文件后，运行
```bash
git add . 
git commit -m  "commit content:"
git push origin main
```
把文件推送给blogsource私有仓库后，github会自动静态部署到lilfry09.github.io上，点开就能看到啦





