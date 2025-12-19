---
created: 2025-12-19 16:27
url: https://code.claude.com/docs/zh-CN/skills
tags:
  - agent
---
## Claude Code Skills

Agent Skills 将专业知识打包成可发现的功能。每个 Skill 包含一个 `SKILL.md` 文件，其中包含 Claude 在相关时读取的说明，以及可选的支持文件，如脚本和模板。

**Skills 如何被调用**：Skills 是**模型调用的**——Claude 根据您的请求和 Skill 的描述自主决定何时使用它们。这与斜杠命令不同，斜杠命令是**用户调用的**（您显式输入 `/command` 来触发它们）。

项目级 Skills:  项目根目录创建`mkdir -p .claude/skills/my-skill-name`
个人 Skills: `mkdir -p ~/.claude/skills/my-skill-name`

```
my-skill/
├── SKILL.md (required)
├── reference.md (optional documentation)
├── examples.md (optional examples)
├── scripts/
│   └── helper.py (optional utility)
└── templates/
    └── template.txt (optional template)
```

在 skill 中引用文件

```
For advanced usage, see [reference.md](reference.md).
```

示例的 Skills https://github.com/anthropics/skills

## Github Copilot Skills

在执行任务时，Copilot 会根据你的提示和技能描述决定何时使用你的技能。

- 创建一个 `.github/skills` 目录来存储你的技能。
- 存储在 `.claude/skills` 目录下的技能也同样受支持。

介绍文档
https://docs.github.com/en/copilot/concepts/agents/about-agent-skills

一个示例的 skill
https://github.com/github/awesome-copilot/tree/main/skills/webapp-testing

skill 文件示例

```md
---
name: github-actions-failure-debugging
description: Guide for debugging failing GitHub Actions workflows. Use this when asked to debug failing GitHub Actions workflows.
---

To debug failing GitHub Actions workflows in a pull request, follow this process, using tools provided from the GitHub MCP Server:

1. Use the `list_workflow_runs` tool to look up recent workflow runs for the pull request and their status
2. Use the `summarize_job_log_failures` tool to get an AI summary of the logs for failed jobs, to understand what went wrong without filling your context windows with thousands of lines of logs
3. If you still need more information, use the `get_job_logs` or `get_workflow_run_logs` tool to get the full, detailed failure logs
4. Try to reproduce the failure yourself in your own environment.
5. Fix the failing build. If you were able to reproduce the failure yourself, make sure it is fixed before committing your changes.

```