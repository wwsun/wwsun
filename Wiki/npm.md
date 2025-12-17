---
created: 2025-12-15 21:14
url: https://docs.npmjs.com/cli/v11/commands
tags:
  - npm
---
## 查看全局包是否需要升级

```bash
npm outdated -g

npm outdated -g --depth=0
```

或者使用 `npm-check`

```bash
# npm i npm-check -g

npm-check -g
```

[https://github.com/dylang/npm-check](https://github.com/dylang/npm-check)

## 添加/更新 tag 到指定版本

```bash
npm dist-tag add <package>@<version> <tag>

npm dist-tag add my-lib@2.0.0-beta.1 beta
```

移除 tag

```bash
npm dist-tag rm <package> <tag>
```

## npm 包加权限

```bash
npm owner add zhousunjing @music/kernel-service-login --registry=http://rnpm.hz.netease.com
```

## 查看包的信息

```bash
npm list              # 查看已安装的包（树形结构）
npm list --depth=0    # 只显示顶层包
npm view <包名>       # 查看包的详细信息
npm info <包名> versions  # 查看包的所有版本
```

## 缓存管理

```bash
npm cache clean --force  # 清理 npm 缓存（问题排查时很有用）
npm cache verify         # 验证缓存完整性
```

## 私有 registry

```bash
# 登陆
npm login --registry <http://localhost:4873/>

# 查看当前用户
npm whoami --registry <http://localhost:4873/>

```