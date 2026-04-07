# 如何用hugo构建自己的博客

# 这篇写给谁

如果你也想搭一个自己的博客，而且希望：

- 源码放在一个仓库里，方便自己慢慢写
- 最终网站自动部署到 `yourusername.github.io`
- 尽量少踩 GitHub Pages 和 Hugo 的坑

那这套流程会比较顺手。

我这里用的是两个仓库：

- 私有仓库：`blogsource`
- 公开仓库：`yourusername.github.io`

其中公开仓库的名字最好直接用 `yourusername.github.io`。如果你是第一次弄 GitHub Pages，建议就按这个来，最省事。

## 开始前先准备好

先安装这几个东西：

- `Hugo`
- `Git`
- 一个你顺手的编辑器，比如 `VS Code`

然后在本地找一个文件夹专门放博客项目，比如我的是 `D:\MyBlog`。

## 一个小提醒

整个流程里，尽量都在本地改文件，再通过 Git 推上去。不要一会儿本地改、一会儿又去 GitHub 网页上改，不然很容易把仓库状态搞乱。

## 整体思路

这套博客的结构其实很简单：

1. `blogsource` 用来放 Hugo 源码
2. `yourusername.github.io` 用来放生成后的静态网页
3. 本地写文章，推送源码
4. GitHub Actions 自动构建并部署到公开仓库

如果你刚开始不想上自动化，也可以先手动部署一遍，确认网站能跑起来，再加 Actions。

## 1. 初始化 Hugo 项目

先在终端里进入你准备放博客的目录，例如：

```powershell
cd D:\MyBlog
```

然后创建 Hugo 项目：

```bash
hugo new site blogsource
cd blogsource
git init
git remote add origin https://github.com/yourusername/blogsource.git
```

这里把 `yourusername` 换成你自己的 GitHub 用户名。

## 2. 添加主题

我这里用的是 `LoveIt` 主题，你也可以换成自己喜欢的主题。

```bash
git submodule add https://github.com/dillonzq/LoveIt themes/LoveIt
```

用子模块的好处是后面更新主题会比较方便。

## 3. 配置 `hugo.toml`

先写一个最基础的配置：

```toml
baseURL = "https://yourusername.github.io/"
title = "My Blog"
theme = "LoveIt"
publishDir = "public"
```

这里有两个地方要特别注意：

- `baseURL` 要换成你自己的地址，而且建议保留最后那个 `/`
- `publishDir = "public"` 表示 Hugo 生成出来的静态文件会放到 `public/`

## 4. 先创建一篇测试文章

先确认 Hugo 本身是正常工作的：

```bash
hugo new posts/hello-world.md
```

然后编辑 `content/posts/hello-world.md`，写成这样：

```markdown
+++
title = "Hello World"
date = "2024-01-01T00:00:00+08:00"
draft = false
+++

这是我的第一篇博客。
```

如果你文章里有 `draft = true`，那默认不会显示出来，这个地方很容易忘。

## 5. 本地预览

在项目根目录运行：

```bash
hugo server -D
```

然后打开：

`http://localhost:1313`

如果你能正常看到首页和刚才那篇测试文章，说明本地 Hugo 已经没问题了。

## 6. 先把源码推到私有仓库

这一步是把 Hugo 源码放进 `blogsource` 仓库。

### 如果远程仓库配错过

可以先删掉旧的：

```bash
git remote remove origin
git remote add origin https://github.com/yourusername/blogsource.git
```

### 提交并推送

```bash
git add .
git commit -m "Initial commit: Hugo project with theme"
git branch -M main
git push -u origin main
```

如果你是一个全新的空仓库，通常这样就够了，不一定非要 `--force`。

推送完成后，可以去 GitHub 上看一眼：

`https://github.com/yourusername/blogsource`

确认文件都已经上去了。

## 7. 先手动部署一次到公开仓库

如果你想先验证整套流程能跑通，可以先手动部署一遍。

先生成静态文件：

```bash
hugo --minify
```

然后进入 `public/`，把它作为一个单独仓库推到公开仓库：

```bash
cd public
git init
git add .
git commit -m "Deploy to GitHub Pages"
git remote add origin https://github.com/yourusername/yourusername.github.io.git
git branch -M main
git push -u origin main --force
```

这里之所以常见 `--force`，是因为 `public/` 这个仓库本质上是生成产物，通常直接整体覆盖最省心。

## 8. 配置 GitHub Pages

去公开仓库 `yourusername.github.io` 里打开：

`Settings -> Pages`

然后确认：

- Branch: `main`
- Folder: `/ (root)`

等一两分钟后，访问：

`https://yourusername.github.io`

如果网站已经能打开，说明最基本的部署链路已经通了。

## 9. 再做自动化部署

等你手动部署成功后，再上 GitHub Actions 会更稳。

目标是：以后你只需要把源码推到 `blogsource`，GitHub 就会自动构建并部署。

## 10. 生成 SSH 密钥

在 `Git Bash` 里运行：

```bash
ssh-keygen -t ed25519 -C "actions-deploy-key" -f ~/.ssh/gh-pages-key
```

生成后有两个文件：

- 私钥：`~/.ssh/gh-pages-key`
- 公钥：`~/.ssh/gh-pages-key.pub`

你可以分别查看：

```bash
cat ~/.ssh/gh-pages-key
cat ~/.ssh/gh-pages-key.pub
```

## 11. 把密钥配置到 GitHub

### 给公开仓库添加公钥

进入 `yourusername.github.io` 仓库：

`Settings -> Deploy keys -> Add deploy key`

然后填写：

- Title: `ACTIONS_DEPLOY_KEY`
- Key: 粘贴 `gh-pages-key.pub` 的内容
- 勾选 `Allow write access`

### 给源码仓库添加私钥

进入 `blogsource` 仓库：

`Settings -> Secrets and variables -> Actions -> New repository secret`

然后填写：

- Name: `ACTIONS_DEPLOY_KEY`
- Value: 粘贴 `gh-pages-key` 的完整内容

## 12. 添加 GitHub Actions 工作流

在项目里新建文件：

`.github/workflows/deploy.yml`

内容可以参考下面这份：

```yaml
name: Deploy Hugo Site to GitHub Pages

on:
  push:
    branches: ["main"]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: true
          fetch-depth: 0

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v2
        with:
          hugo-version: "0.145.0"
          extended: true

      - name: Build
        run: hugo --minify --gc

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          deploy_key: ${{ secrets.ACTIONS_DEPLOY_KEY }}
          external_repository: yourusername/yourusername.github.io
          publish_dir: ./public
          publish_branch: gh-pages
          keep_files: false
```

这里也记得把 `yourusername` 换成你自己的用户名。

## 13. 验证工作流是否生效

把工作流文件提交上去：

```bash
git add .github/workflows/deploy.yml
git commit -m "Add GitHub Actions workflow"
git push origin main
```

然后去 GitHub 的 `Actions` 页面看：

- 有没有出现新的 workflow
- 最新一次运行是不是绿色
- 如果失败了，点进去直接看日志

## 14. 后面写文章就会轻松很多

当自动化部署配好以后，后续更新博客基本就是这套流程：

1. 在 `content/` 里写或修改文章
2. 本地预览确认没问题
3. 提交源码
4. 推送到 `blogsource`
5. 等 GitHub 自动部署完成

命令大概就是：

```bash
git add .
git commit -m "Update blog content"
git push origin main
```

推上去之后，GitHub 会自动把静态页面部署到 `yourusername.github.io`。

## 最后补一句

如果你只是想先把博客跑起来，那就先完成下面这几步：

1. 创建 Hugo 项目
2. 加主题
3. 本地预览
4. 手动部署成功一次

等这条链路走通以后，再加自动化，心态会轻松很多，也更不容易被一堆 Git 和 Actions 报错劝退。




