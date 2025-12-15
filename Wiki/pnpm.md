---
created: 2025-12-15 09:49
url: https://pnpm.io/zh/feature-comparison
tags:
  - pnpm
  - monorepo
---
## pnpm add pkg-name

```bash
# save to dependencies
pnpm add sax

# save to devDependencies
pnpm add -D sax
```

## pnpm update/up

```bash
pnpm up

pnpm up -i
```

https://pnpm.io/cli/update

## workspace
monorepo support
https://pnpm.io/zh/workspaces

### `pnpm-workspace.yaml`

```
packages:
  # 指定根目录直接子目录中的包
  - 'my-app'
  # packages/ 直接子目录中的所有包
  - 'packages/*'
  # components/ 子目录中的所有包
  - 'components/**'
  # 排除测试目录中的包
  - '!**/test/**'
```

