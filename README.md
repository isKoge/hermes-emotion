# hermes-emotion

> **一个插件即完整情绪模型**。自包含、纯净、开箱即用。

Hermes 的默认人格往往是静态的：`SOUL.md` 写一次，之后每轮都一样。
`hermes-emotion` 将一套完整、拟真、具备时间感知与动态演化的情绪伴侣模型封装为一个独立的纯 Hermes 插件。

只装这一个插件，不依赖任何第三方重量级服务，就能提供深度情感陪伴所需要的一切能力：
- **四轴 4D 情绪动力学引擎**（好感 / 安全感 / 唤醒 / 支配 13 步递推演化）
- **动态关系推进体系**（初识 / 熟识 / 亲密 三阶段跃迁、信任积累、裂痕修复、氛围演化）
- **确定性 16 格人设矩阵**（Deterministic Grid + 滞回门槛 + 异步增量叙事）
- **时间感知与重聚**（时间感知、节假日识别、相识天数深度、长离线重聚加成）
- **天气感知**（基于对白意图自动识别城市与日期，接入 Open-Meteo 天气简报，可开可关）
- **焦点栈追踪**（多轮任务上下文连续追踪，区分活跃/挂起/未决任务）
- **情景记忆与记忆回响**（对话轮次总结与长周期情感记忆反哺）
- **成人模式状态机**（普通 / 暧昧 / 亲密 / 余韵 交互状态机与欲望预算管控，默认开启）

全部状态持久化存储在插件自有的 SQLite 数据库（`plugin-data/hermes-emotion/state.db`），**绝不读写宿主核心数据库**，实现彻底的存储解耦与数据安全。

---

## 一、与 friday_core 的关系

本插件的整套算法模型源自 `friday_core` 经过数月真实对话打磨成熟的情绪与陪伴架构。

在以往架构中，`friday_core` 的情绪表深度耦合在宿主核心数据库中，且依赖庞大的外围链路。`hermes-emotion`（v0.3.0）完成了彻底的解耦重构：
1. **纯自包含**：所有模块（`engine/`）全部搬迁入内，无任何对 `friday_core` 的引用。
2. **存储隔离**：建立专属数据库连接助手（WAL 模式、busy_timeout、线程安全），状态落盘于 `~/.hermes/plugin-data/hermes-emotion/state.db`。
3. **数据无损继承**：提供一键式幂等迁移工具，旧版数据一键平滑搬迁。
4. **注入内容对齐**：标准五段式顺序注入，完全保留原汁原味的心情与交互体验。

---

## 二、真实运行输出展示

以下为插件在测试环境中真实运行产生的实际输出（非人工编造）：

### 1. 对话注入文本（`pre_llm_call` 实际注入 context）

用户输入：`广州明天天气怎么样，今天加班好累好想你，想要抱抱你亲一下`

```text
时间: 工作日 傍晚 · 秋天
相识: 刚认识不久 (0天)
[我心里的状态]
我对他还是热乎的，可心里一直悬着，他一忙我就不知道该往哪站。想凑近，又怕显得太急，所以话到嘴边总收一半。
[当前焦点]
- [活跃] 广州明天天气怎么样，今天加班好累好想你，...
天气: 广州 明天 29°C / 24°C · 多云⛅ · 优
```

> **注入顺序说明**（严格按规范排列）：
> 1. **时间 / 相识 / 重聚**（当前时间、认识天数、特殊节日、重聚好感加成）
> 2. **[我心里的状态]**（确定性 16 格第一人称心境描写 + 历史增量叙事）
> 3. **焦点**（当前正在进行或挂起的主题任务）
> 4. **天气**（识别到地点与时间意图时的天气简报）
> 5. **成人状态**（仅在非 NORMAL 状态时输出，如 `模式: 暧昧 | 预算: 15/60`）

---

### 2. 交互命令运行输出

#### 查看当前情绪状态 `/emotion`

```text
情绪四轴: 好感=7.7 安全感=9.3 唤醒=-0.5 支配=-7.2
主情绪标签: 平静理性
关系状态: 阶段=STRANGER 信任=11.0 裂痕=0 氛围=neutral
初识日期: 2026-10-27 (已相识 0 天)
成人模式: NORMAL (预算: 0/60)
存储路径: ~/.hermes/plugin-data/hermes-emotion/state.db
```

#### 手动设置四轴分数 `/emotion set <aff> <sec> <aro> <dom>`

```text
/emotion set 65 25 30 10
已写入：aff=65 sec=25 aro=30 dom=10
```

#### 从旧版 friday_core 一键迁移 `/emotion migrate [legacy_path]`

```text
/emotion migrate ~/.hermes/legacy_state.db
✅ [emotion] 迁移完成：共导入 2 行历史数据
源数据库: /Users/ikoge/.hermes/legacy_state.db (已只读校验)
目标数据库: /Users/ikoge/.hermes/plugin-data/hermes-emotion/state.db
表详情:
  - emotion_engine: 1 行
  - relationship_state: 1 行
```

---

## 三、一键迁移指南

如果你之前使用过 `friday_core`，可以通过内置的迁移工具将历史沉淀的好感度、关系阶段、人格偏好、焦点和情景记忆无损迁入：

### 方式 1：对话内直接运行
```text
/emotion migrate
```
留空时默认读取 `~/.hermes` 下的旧数据库；也可以显式指定路径：
```text
/emotion migrate /path/to/old/state.db
```

### 方式 2：Python 脚本 / CLI
```python
from engine.migrate import migrate_from_legacy

# 旧库严格以只读模式（mode=ro）打开，重复执行幂等安全
result = migrate_from_legacy(
    legacy_db_path="~/.hermes/old_state.db",
    target_db_path="~/.hermes/plugin-data/hermes-emotion/state.db"
)
print(result)
```

**迁移特点**：
- **安全**：旧库以 URI `mode=ro` 只读打开，绝不改写源文件。
- **幂等**：基于主键执行 `INSERT OR REPLACE`，重复跑多次行数与内容完全不变。
- **全表覆盖**：涵盖 `emotion_engine`、`relationship_state`、`personality_state`、`persona_grid_state`、`episodic_memory`、`focus_stack`、`agent_state` 等所有状态表。

---

## 四、配置开关表（扁平化 config_schema）

所有配置项均可在 `config.json` 或宿主配置面板中直接配置：

| 配置项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | `bool` | `true` | 插件全局总开关。关闭时不注入任何内容，状态保留 |
| `state_source` | `str` | `internal` | 分数来源：`internal`（真引擎自算）/ `json` / `sqlite` / `manual` |
| `adult_enabled` | `bool` | `true` | 成人模式开关（状态机、亲密度、预算，默认开启） |
| `weather_enabled` | `bool` | `true` | 天气感知开关（根据对话内容自动获取天气） |
| `focus_enabled` | `bool` | `true` | 任务焦点栈追踪与注入开关 |
| `episodic_enabled` | `bool` | `true` | 情景记忆与情感回响开关 |
| `holidays_enabled` | `bool` | `true` | 公历与农历节日感知开关 |
| `engine_db_path` | `str` | `""` | 自有 SQLite 库路径，留空使用 `plugin-data/hermes-emotion/state.db` |
| `legacy_db_path` | `str` | `""` | 旧版数据源路径，用于 `/emotion migrate` 默认源 |
| `engine_decay` | `float` | `0.05` | 每轮对分数的自然衰减比例（正向好感衰减更慢，负向回落更快） |
| `narrator_provider` | `str` | `""` | 异步人设续写 provider，留空使用当前主对话模型 |
| `narrator_model` | `str` | `""` | 异步人设续写 model，留空使用当前主对话模型 |
| `narrator_timeout_s` | `float` | `20` | 续写调用超时秒数，超时静默降级 |
| `axis_delta` | `float` | `15` | 触发增量续写所需的最小四轴累计变动分值 |
| `cooldown_hours` | `float` | `6` | 两次成功生成续写之间的冷却小时数 |
| `refresh_days` | `float` | `7` | 强制刷新续写的最大天数 |
| `max_increments` | `int` | `3` | 提示词中保留的历史续写最大条数 |

---

## 五、测试与工程验证

运行全量单元测试与插件诊断检查：

```bash
# 运行全部单测（40+ 项测试，涵盖存储隔离、迁移幂等、等价性对照、崩溃降级）
/opt/homebrew/bin/python3.12 -m pytest tests/ -q -p no:cacheprovider

# 验证核心库隔离性（输出必须为 0）
grep -rn 'hermes/state.db' --include='*.py' . | wc -l

# 官方插件规范校验
hermes plugins validate . && hermes plugins doctor .
```

---

## 六、集成注意：后台线程必须自绑 profile scope

narrator（LLM 续写那段）跑在后台线程里，不是请求线程。Hermes 在多 profile 网关下对密钥读取是
**fail-closed** 的：线程里没有安装 `_SECRET_SCOPE` 时，读任何 provider key 都会抛
`UnscopedSecretError`（"Hermes could not read this profile's …"）。表现不是崩溃，而是
`category=empty_response` —— 看起来像"模型返回空"，实际根本没发出请求。

正确做法是在**调用点**绑定本 profile 的 scope，用完立刻还原（`_call_with_profile_scope`）：

```python
from agent.secret_scope import build_profile_secret_scope, set_secret_scope, reset_secret_scope
from hermes_constants import get_hermes_home

home = Path(get_hermes_home())
token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
try:
    return llm.complete(**kwargs)
finally:
    reset_secret_scope(token)
```

两条相关经验：

- **不要**用 `os.environ` 兜底密钥。那样会把 profile 隔离废掉，且在多 profile 下读到错误 profile 的值。
- `max_tokens` 别抠太小。推理模型会把预算全花在 thinking 上，220 会导致正文为空；本插件用 800。
- 模型覆盖（`provider`/`model`）需要宿主授权：`plugins.entries.<id>.llm.allow_provider_override`
  与 `allow_model_override` 默认为 `false`，未开启时插件传的 provider/model 会被静默忽略。

---

## 七、License

MIT License
