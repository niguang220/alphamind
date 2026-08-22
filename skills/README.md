# AlphaMind Skills 文档

AlphaMind 启动时会从 `ALPHAMIND_SKILLS_DIR` 读取 Skills，并在匹配用户请求时注入到对应 Agent 的 system prompt。Skills 适合维护业务处理规范、投研问答口径、合规话术、适当性与升级边界和禁止事项。

当前内置三类 Skills：

```text
skills/market_info/SKILL.md            # 行情与信息：行情/指数/ETF信息/术语解释，客观中立
skills/research_analysis/SKILL.md      # 投研与分析：研报/财报/估值/量化概念，只解读不下结论
skills/compliance_suitability/SKILL.md # 合规与适当性：风险等级/风险揭示/开户/交易规则，守合规底线
```

## Skill 文件格式

推荐每个 Skill 使用独立目录，并将主文件命名为 `SKILL.md`：

```text
skills/<skill_name>/SKILL.md
```

文件顶部使用简单 front matter：

```markdown
---
name: 投研与分析规范
description: 适用于 ResearchAgent 的研报/财报/估值/量化概念解读规范
keywords: 研报,财报,估值,市盈率,因子,回测,夏普,基本面
agents: research
enabled: true
---
```

字段说明：

- `name`：Skill 展示名称，会出现在注入给模型的 prompt 中。
- `description`：简短说明，方便 `/skills` 接口排查。
- `keywords`：触发关键词，用户消息命中后才注入；多个关键词用英文逗号或中文逗号分隔均可。
- `agents`：适用 Agent，可填 `market`、`research`、`compliance`，多个值用逗号分隔。
- `enabled`：是否启用，支持 `true/false`。

## 编写要求

- 重要规则放在文档前半部分，因为过长内容会按 prompt 预算截断。
- 一类 Skill 只描述一类职责，不要把行情信息、投研分析、合规适当性规则混在一个文件里。
- 必须包含“角色定位”“处理流程”“升级条件”“禁止事项”等稳定章节。
- 对用户隐私、支付、密码、验证码、API Key、Token 等敏感信息必须写明禁止收集或禁止公开。
- 对无法保证的事项使用保守措辞，例如“通常”“预计”“需要核验后确认”。
- 对需要人工、财务、二线技术处理的场景要明确写出升级条件。

## 热加载

修改 Skill 文件后，不需要重启服务，调用：

```bash
curl -X POST http://localhost:8000/skills/reload
```

查看加载结果和解析错误：

```bash
curl http://localhost:8000/skills
```
