---
title: 搭建博客流程
categories: 工程
tags:
- 环境配置
- blog
- hexo
---


## Quick Start

### 首先你需要准备npm 、nodejs开发工具

``` python
apt update && apt upgrade
apt install npm nodejs
```



### 初始化hexo项目

``` bash
npm install hexo-cli -g
mkdir myblog && cd myblog
hexo init
npm install
```

### 选择主题

从hexo官方[主题仓库](https://hexo.io/themes/)中选择自己喜欢的主题即可！

### 部署在github pages上

新建 git 仓库，名称为：.github.io (好处是：通过https://username.github.io/就可以访问到你的仓库/blog/project主页, 而不需要在 github.io/后面再加上仓库名)

安装插件：npm install hexo-deployer-git --save

创建一个 SSH key：ssh-keygen -t -rsa -C "email address"

将密钥添加到 github，验证是否添加成功：ssh -T git@github.com



``` bash
# 在 _config.yml 文件最后修改为：
deploy:
  type: git
  repo: git@github.com:LikeFrost/LikeFrost.github.io.git
  branch: master

  then
hexo deploy
```

