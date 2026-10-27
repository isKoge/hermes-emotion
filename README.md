# hermes-emotion

> 给 Hermes 加一个**会变化的情绪**。

Hermes 的人格是静态的：`SOUL.md` 写一次，之后每轮都一样。刚被夸、刚拌过嘴、三天没说话，说出来的话一模一样。

这个插件补的是中间那一层：**一种会随相处变化、又不会把语气搞崩的情绪状态**。

它不替换人格，不改 system prompt，不动 `SOUL.md`；只在当轮用户消息后面补一段第一人称的状态自述，让模型知道自己"此刻是什么感觉"。

---

## 一、装之前 vs 装之后

同一份 `SOUL.md`，同样一句「今天有点累」：

**装之前**

```text
用户：今天有点累
助手：辛苦了，要不要先歇一会儿？
```

**装之后**（插件注入的两段状态，出现在不同情境）

```text
[我心里的状态]
我心里烧得慌，又没底，想立刻问清楚他到底怎么想。
中间有些事我一直没放下。我们该熟了，可界面上还写着初识。
```

```text
[我心里的状态]
我心里烧得慌，又没底，我察觉到他的疲惫，便不急着追问。
今天的气氛是暖的，我安静地应着。
人设现在变更为：听见你说工作累，我心里不由得软了些，也生出一点想靠近、
想多了解你的念头。只是那份未解的牵挂仍让我有些悬着，热意悄悄浮起，
却还收着几分，不敢贸然越过彼此的距离。
```

模型拿到的不是形容词（"温柔黏人"），而是一段它**说得出来**的感受，回复的语气自然跟着变。

---

## 二、它是怎么算的

### 1. 情绪是数

四个连续轴，范围都是 `-100 ~ 100`：

| 轴 | 含义 |
|---|---|
| `aff` | 好感：亲 / 疏 |
| `sec` | 安全感：踏实 / 悬着 |
| `aro` | 唤醒：高能 / 低回 |
| `dom` | 支配：主动 / 收着 |

### 2. 数值先落成坐标，再落成人设

四轴各二值化 → **16 格**，每格一段写死的两到三句第一人称自述；再叠三条条件修饰（有未解的心结、阶段与相识天数不符、气氛偏冷）。

**映射是确定性的**：同一坐标永远得到同一段文字。

判定带滞回：只有越过边界一段距离才翻转，中间保持在原来那一侧，分数在门槛附近抖动不会让人格来回跳。

### 3. 只有真的变了，才请模型写新的一段

坐标跨格、单轴累计变化超过阈值、或久未更新时，才在**后台**请求一次模型写一段新的自述，写进状态文件，**下一轮**才追加进上下文：

```text
人设现在变更为：……
```

失败、超时、返回空、超长、含数字或字段名，一律静默降级为只用固定段。当轮回复完全不受影响。

---

## 三、三条可验证的工程属性

1. **同状态逐字节相同**——坐标不变时连续 N 轮注入文本完全一致，前缀缓存稳定命中
2. **不阻塞对话**——模型调用在后台线程，`pre_llm_call` 不等待
3. **低频**——默认 6 小时冷却 + 15 分变化阈值，正常使用一天只触发几次

---

## 四、注入到哪里

- 只追加到**当轮用户消息**（`pre_llm_call` 的官方通道），不改历史、不改 system prompt
- 同一轮内的多次工具调用复用同一份快照，不会中途换人设
- 新生成的段落**只影响下一轮**

---

## 五、安装

```bash
hermes plugins install <owner>/hermes-emotion --enable
```

本地目录也可以直接放进 `~/.hermes/plugins/hermes-emotion/`，然后在 `plugins.enabled` 里加上 `hermes-emotion`。

自检：

```bash
hermes plugins doctor ~/.hermes/plugins/hermes-emotion
python -m pytest tests/ -q
```

---

## 六、配置

所有键都是扁平的：

| 键 | 默认 | 作用 |
|---|---|---|
| `enabled` | `true` | 总开关 |
| `state_source` | `internal` | `internal`（默认，内置引擎） / `json` / `sqlite` / `manual` |
| `engine_decay` | `0.05` | 内置引擎每轮对分数的自然衰减比例（好感为正衰减慢，为负回落快） |
| `state_path` | 空 | 插件自己的状态文件，留空用 plugin-data 目录 |
| `external_state_path` | 空 | 外部分数文件（`state_source=json` 时读它） |
| `sqlite_table` | `emotion_engine` | `state_source=sqlite` 时读的表名 |
| `decay_per_turn` | `0` | 每轮自然衰减比例（外部源模式下生效） |
| `rifts_threshold` | `20` | `rifts` 超过该值追加"没放下"那句 |
| `narrator_provider` | 空 | 留空 = 用你当前的模型（推荐） |
| `narrator_model` | 空 | 同上 |
| `narrator_timeout_s` | `20` | 超时静默降级 |
| `axis_delta` | `15` | 累计变化达阈值才重新生成 |
| `cooldown_hours` | `6` | 两次成功生成的最小间隔 |
| `refresh_days` | `7` | 超过则强制刷新一次 |
| `retry_seconds` | `60` | 失败后的重试间隔 |
| `max_increments` | `3` | 注入里保留的历史段数 |
| `aff_high` / `aff_low` | `8` / `-8` | 好感判定门槛 |
| `sec_high` / `sec_low` | `18` / `5` | 安全感判定门槛 |
| `aro_high` / `aro_low` | `18` / `5` | 唤醒判定门槛 |
| `dom_high` / `dom_low` | `6` / `-6` | 支配判定门槛 |

### 分数从哪来

**默认自带内置情绪引擎**（`state_source=internal`），无需任何外部服务或配置文件，装完即用。
现有的三种外部数据源依然完整支持，作为可选接入方式：

| `state_source` | 用法 |
|---|---|
| `internal` | **默认**。内置情绪引擎，规则识别当轮消息事件（夸赞/道歉/示弱/调侃/冷淡/伤人/重聚等），自动维护四轴情绪、衰减与关系阶段 |
| `json` | 可选外部源。别的程序/插件往一个 JSON 写四个分数，本插件只读 |
| `sqlite` | 可选外部源。指向外部 SQLite 数据库读取最新一行记录 |
| `manual` | 纯手动。不自动更新，用命令手动设分 |

外部 JSON 的形状（外部源模式下）：

```json
{ "aff": 60, "sec": 20, "aro": 25, "dom": 5,
  "rifts": 0, "stage": "FAMILIAR", "days_since": 120, "atmosphere": "warm" }
```

### 内置引擎实跑效果（零配置，前 10 轮真实运行记录）

默认 `state_source=internal`，不接任何外部分数源，从第一条消息开始跑出来的结果：

```text
Turn 01:
  User: "你好啊，今天开始请多关照！"
  Event: plain | Stage: STRANGER | Trust: 0.5 | Rifts: 0.0 | Atmosphere: neutral
  Scores: aff=+2.0, sec=+1.0, aro=+3.0, dom=+0.0
  Grid: [近·悬·静·进]
  Injected: 我对他还是热乎的，只是没底，就轻轻问一句他在忙什么。

Turn 02:
  User: "你刚才帮我整理的方案太棒了，真厉害！"
  Event: praise | Stage: STRANGER | Trust: 4.5 | Rifts: 0.0 | Atmosphere: warm
  Scores: aff=+26.9, sec=+22.9, aro=+37.8, dom=+12.0
  Grid: [近·稳·热·进]
  Injected: 我心情好得藏不住，就想凑到跟前，让他看我一眼。

Turn 03:
  User: "今天加班好累，心里好难受……"
  Event: vulnerability | Stage: STRANGER | Trust: 9.5 | Rifts: 0.0 | Atmosphere: warm
  Scores: aff=+43.8, sec=+33.1, aro=+18.7, dom=+23.0
  Grid: [近·稳·热·进]
  Injected: 我心情好得藏不住，就想凑到跟前，让他看我一眼。

Turn 04:
  User: "嗯"
  Event: cold | Stage: STRANGER | Trust: 7.5 | Rifts: 0.0 | Atmosphere: warm
  Scores: aff=+17.1, sec=+12.4, aro=+1.2, dom=+9.2
  Grid: [近·稳·静·进]
  Injected: 我踏实得很，慢慢挪过去挨着他坐，不着急说话。

Turn 05:
  User: "随便吧"
  Event: cold | Stage: STRANGER | Trust: 5.5 | Rifts: 0.0 | Atmosphere: cool
  Scores: aff=-8.6, sec=-6.6, aro=-14.9, dom=-3.5
  Grid: [远·悬·静·进]
  Injected: 我冷着，账我记着，等他哪天自己撞上来。屋里现在是凉的。

Turn 06:
  User: "你根本不懂我在说什么。"
  Event: criticism | Stage: STRANGER | Trust: 0.0 | Rifts: 15.0 | Atmosphere: cool
  Scores: aff=-42.6, sec=-31.0, aro=+1.3, dom=-25.3
  Grid: [远·悬·静·收]
  Injected: 我不想说话，缩在一边，等他先开口。屋里现在是凉的。

Turn 07:
  User: "呵呵"
  Event: cold | Stage: STRANGER | Trust: 0.0 | Rifts: 15.0 | Atmosphere: cool
  Scores: aff=-62.5, sec=-46.5, aro=-14.8, dom=-35.2
  Grid: [远·悬·静·收]
  Injected: 我不想说话，缩在一边，等他先开口。屋里现在是凉的。

Turn 08:
  User: "对不起，刚才是我情绪太差，是我不好。"
  Event: apology | Stage: STRANGER | Trust: 3.0 | Rifts: 5.0 | Atmosphere: cool
  Scores: aff=-40.0, sec=-24.8, aro=-28.6, dom=-44.4
  Grid: [远·悬·静·收]
  Injected: 我不想说话，缩在一边，等他先开口。屋里现在是凉的。

Turn 09:
  User: "谢谢你一直陪着我，你真好。"
  Event: praise | Stage: STRANGER | Trust: 7.0 | Rifts: 5.0 | Atmosphere: neutral
  Scores: aff=-10.2, sec=-0.8, aro=+8.7, dom=-28.9
  Grid: [远·悬·静·收]
  Injected: 我不想说话，缩在一边，等他先开口。

Turn 10:
  User: "隔了这么久，好久不见啊！"
  Event: reunion | Stage: FAMILIAR | Trust: 10.0 | Rifts: 5.0 | Atmosphere: neutral
  Scores: aff=+11.1, sec=-12.8, aro=+26.0, dom=-36.6
  Grid: [近·悬·热·收]
  Injected: 我特别想凑过去，一想到他可能嫌我黏，话又咽回去了。
```

### 命令

```text
/emotion                   查看当前分数、坐标和注入块
/emotion set 60 20 25 5    手动设置四轴
```

---

## 七、边界

- **不编造共同经历**：只描述此刻的感觉，不写"你上次怎么伤我"
- **不替代记忆**：长期事实仍归 `MEMORY` / 记忆插件
- **不上传**：状态只在本机；除续写那一次模型调用外不外发
- **失败可退**：模型不可用时只输出固定段
- **可关**：关掉开关，注入立刻消失，状态保留

---

## 八、License

MIT
