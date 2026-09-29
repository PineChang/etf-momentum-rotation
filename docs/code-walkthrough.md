# 项目代码完全导读:ETF 三因子动量轮动策略

> 面向想读懂、改懂这套代码的学习者。
> 逐模块讲清楚:**它负责什么、怎么实现的、为什么这样设计**,最后讲部署与调用。
> 全文约 2000 行 Python(含测试),核心引擎约 700 行,是一个"小而完整"的量化项目范本。
>
> ⚠️ **本文的代码行号锚点写于早期版本,重构后已整体漂移——请以函数/类名为准,不要按行号跳转。**
> 第四轮(2026-09-29)的修复清单见 `docs/audit-report.md` §九;当前数字见 `README.md`。

## 目录

1. [项目全景:数据怎么流动](#一项目全景数据怎么流动)
2. [config.py — 所有旋钮集中在一处](#二configpy--所有旋钮集中在一处)
3. [factors.py — 三个动量因子(策略的大脑)](#三factorspy--三个动量因子策略的大脑)
4. [scoring.py — 标准化、加权与选股](#四scoringpy--标准化加权与选股)
5. [data.py — 数据获取与缓存](#五datapy--数据获取与缓存)
6. [backtest.py — 回测引擎(项目的心脏)](#六backtestpy--回测引擎项目的心脏)
7. [metrics.py / report.py — 绩效评估与可视化](#七metricspy--reportpy--绩效评估与可视化)
8. [live.py — 实盘桥(盘前计划与状态持久化)](#八livepy--实盘桥盘前计划与状态持久化)
9. [sweep.py / cli.py — 参数遍历与命令行](#九sweeppy--clipy--参数遍历与命令行)
10. [测试体系:102 个测试保什么](#十测试体系102-个测试保什么)
11. [部署与调用](#十一部署与调用)
12. [设计原则总结(可迁移到其他项目)](#十二设计原则总结可迁移到其他项目)

---

## 一、项目全景:数据怎么流动

```
                        ┌─────────────────────────────────────────┐
                        │            config.py (参数中心)           │
                        └──────────────┬──────────────────────────┘
                                       │ 被所有模块读取
   akshare/腾讯行情 ──► data.py ──► DataFrame(date/open/high/low/close)
                                       │
                                       ▼
                    factors.py (三因子) ──► scoring.py (Z-Score+加权+阈值)
                                       │ 每个交易日收盘后产出一个目标持仓集合
                                       ▼
                    backtest.py (事件循环:开盘执行,收盘出信号)
                                       │ BacktestResult(equity/trades/scores/holdings)
                                       ▼
              metrics.py (绩效指标)  +  report.py (净值图/交易明细)

   实盘路径(与回测共用因子与评分):
                    data.py ──► factors/scoring ──► live.make_plan (交易计划)
                                                       │
                                              live.execute_plan (按成交价执行)
                                                       │
                                              StateStore (JSON 持久化,重启恢复)
```

关键架构决策:**因子与评分是同一份代码,回测和实盘共用**。
回测引擎和实盘桥(`live.py`)不复制策略逻辑,只是给同一套"评分→目标"的决策
接上不同的执行环境(历史数据模拟撮合 vs 真实下单)。这从根本上避免了
"回测很美、实盘代码却是另一套"的经典翻车。

## 二、config.py — 所有旋钮集中在一处

46 行,两类东西:

**1. 股票池**(`config.py:8`):四只 ETF 的字典,代码→名称。

**2. `StrategyConfig`(`config.py:17`)**:一个 `@dataclass(frozen=True)`,全部策略参数:

```python
@dataclass(frozen=True)
class StrategyConfig:
    momentum_day: int = 25            # 文章给出的乖离回归窗口
    w_bias: float = 0.3               # 文章给出的三因子权重
    ...
    rebalance_threshold: float = 1.5  # 文章给出的调仓阈值
    # —— 风控扩展(本项目改良,默认关闭 = 忠实原文)——
    min_momentum: float | None = None
    trailing_stop: float | None = None
    n_hold: int = 1
    ...
```

设计要点:

- **frozen=True(不可变)**:配置创建后不能改,杜绝"回测跑到一半参数被改"的隐患。
  要变体就用 `dataclasses.replace(cfg, stop_loss=0.06)` 生成新对象——
  实验矩阵里几十个变体都是这样派生的;
- **默认值 = 原文策略**:所有风控开关默认 `None`/1(关闭),
  `StrategyConfig()` 空手实例化出来的就是 100% 忠于文章的策略。
  改良是"显式打开开关",不是"改掉默认";
- **`warmup_bars` 属性**(`config.py:45`):算出因子需要的最少历史 bar 数
  (`max(ma_n + momentum_day, slope_n, efficiency_n)`),
  回测引擎靠它知道从第几天开始才能出信号。

## 三、factors.py — 三个动量因子(策略的大脑)

113 行,4 个公开函数 + 1 个数学工具。策略思想的全部"含金量"在这里。

### 3.1 `_ols_slope_r2`(factors.py:32)— 为什么不用 sklearn

文章原文用 `sklearn.linear_model.LinearRegression`,本项目用 numpy 手写一元 OLS:

```python
ss_xy = ((x - x.mean()) * (y - y.mean())).sum()
ss_xx = ((x - x.mean()) ** 2).sum()
slope = ss_xy / ss_xx                    # 最小二乘闭式解
r_squared = 1 - ss_res / ss_tot          # 决定系数
```

一元线性回归的斜率有解析解(协方差/方差),与 sklearn 数值上**完全等价**
(测试里用 `np.polyfit` 独立验证到 1e-9 精度),换来的是零重依赖。
**经验:依赖一个大库之前,先问它的核心公式是不是十行能写完。**

### 3.2 三个因子函数

| 函数 | 文章公式 | 输入 | 输出语义 |
|------|---------|------|---------|
| `bias_momentum_score` (factors.py:49) | 乖离率序列归一化后回归,斜率×10000 | 收盘价 + 两个窗口 | 偏离在加剧吗 |
| `slope_momentum_score` (factors.py:64) | 归一化价格回归,10000×斜率×R² | 收盘价 + 窗口 | 涨得多×涨得稳 |
| `efficiency_momentum_score` (factors.py:75) | 中枢价对数动量×效率系数 | OHLC + 窗口 | 涨得流畅吗 |

以效率因子看实现细节(factors.py:75):

```python
pivot = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4.0).to_numpy(dtype=float)
log_pivot = np.log(pivot)
momentum = 100.0 * (log_pivot[-1] - log_pivot[0])       # 区间对数收益
direction = abs(log_pivot[-1] - log_pivot[0])           # 净位移
volatility = float(np.abs(np.diff(log_pivot)).sum())    # 总路程
efficiency_ratio = direction / volatility if volatility > 0 else 0.0
return momentum * efficiency_ratio
```

注意两个防御细节:① `volatility > 0` 的除零保护(完全横盘的标的);
② 数据不足窗口时返回 `NaN` 而不是硬算——**宁可缺失,不可造假**,由上游决定怎么处理。

### 3.3 `compute_factor_scores`(factors.py:93)— 横截面汇总

对股票池每只 ETF 各算三个因子,返回 `index=symbol, columns=[bias, slope, efficiency]`
的 DataFrame。它是"单标的时序计算"到"多标的横截面比较"的接缝。

**尺度不变性**(重要性质,后面验证会用到):三个因子内部都做了归一化
(除首值/对数收益),所以因子得分与价格的**绝对量级无关**——
前复权、后复权、不同数据源的复权基准不同,因子结果不变。

## 四、scoring.py — 标准化、加权与选股

80 行,四个函数,对应文章第 2、3 节。

### 4.1 `zscore`(scoring.py:15)— 横截面标准化

```python
mean = factor_df.mean(axis=0)
std = factor_df.std(axis=0, ddof=0)
std = std.where(std > 0, other=1.0)    # 防除零:全部相同则标准化后全 0
return (factor_df - mean) / std
```

- 对每个**因子列**(不是每只 ETF)在 4 只 ETF 上做标准化 → "你在池子里排什么位置";
- `ddof=0`(总体标准差):池子只有 4 只,样本/总体差异可忽略,取确定性的那个;
- 单标的池 std=0 时兜底为 1 → 得分全 0,这是后面单标的测试场景的关键行为。

### 4.2 `combined_score`(scoring.py:27)

一行加权:`0.3×z(乖离) + 0.3×z(斜率) + 0.4×z(效率)`,返回每只 ETF 的综合得分。

### 4.3 `select_target` / `select_targets`(scoring.py:33 / 53)— 阈值选股

单持仓版(文章原文逻辑):

```python
if current is None:      return best          # 空仓 → 买第一
if best == current:      return current       # 已是第一 → 持有
if scores[best] > scores[current] * threshold: return best   # 显著超越 → 换
return current                                  # 否则不动
```

多持仓版 `select_targets` 是同一思想的自然推广:

1. 先按排名把 n 个槽位填满(**空槽直接补位**,无需过阈值);
2. 再循环做"换血":挑战者须**显著强于**在持最弱者,才替换。

边界语义都被测试锁死:**恰等于 1.5 倍不换**(文章说"超过")、全 NaN 时维持现状。

> ⚠️ 阈值比较在**负分区**不是一个"照抄文章"的选择,而是踩过的坑:文章的字面写法
> `挑战者 > 在持者 × 1.5` 在在持者得分为负时**反向放行**(`-0.9 > -1.5` 成立),
> "防换血"机制失效,实测 1901 次比较中有 54 次换到了**评分更低**的标的。
> 现改为 `_beats`:挑战者须超出 `|在持者| × (阈值-1)`——正分区与文章公式逐位等价,
> 负分区同样要求显著超出。见 `docs/audit-report.md` §六 R1。

## 五、data.py — 数据获取与缓存

74 行,解决"数据从哪来、怎么不变慢、怎么不怕断"三件事。

### 5.1 分层设计

```
fetch_universe (data.py:66)     批量入口,逐个拉取 + sleep(1) 防限流
    └─ fetch_etf (data.py:38)   单只:先读 CSV 缓存,过期则增量更新,没有再全量
        └─ _fetch_em (data.py:18)  akshare 东财接口,失败重试 4 次(指数退避)
        └─ _normalize (data.py:32)  中文列名→英文,统一 date/open/high/low/close/volume
```

### 5.2 缓存策略(三个实战教训)

```python
cache = DATA_DIR / (f"{symbol}.csv" if adjust == "qfq" else f"{symbol}.{adjust}.csv")
```

- **增量更新**:缓存存在且最后日期距今 ≤4 天(容忍周末/小长假)直接用;否则带 **9 天重叠区**
  增量拉取,把重叠区收盘价与缓存比对——**前复权在分红后会改写整条历史**,锚点变了就必须
  丢弃缓存全量重拉,否则会在除息日留下假跳空(后复权锚定上市日,不受影响);
- **复权方式进文件名**:qfq(前复权)保持旧文件名兼容已有缓存,hfq 单独存。
  来源:一次重构改了缓存名,导致全量重新下载、触发东财限流封 IP;
- **qfq vs hfq**:前复权除息后改写历史(轻微未来信息),后复权不改写。
  策略回测用 qfq(连续直观),严谨性验证用 hfq 复跑(`scripts/validate_robustness.py`)。

### 5.3 备用数据源

`scripts/validate_robustness.py` 里的 `fetch_tx` 演示了腾讯行情接口
(东财限流时的备胎,分页 640 bar/次)。因子尺度不变(见 3.3),跨源结论可比。

## 六、backtest.py — 回测引擎(项目的心脏)

160 行。看懂这一个文件,就看懂了整个项目。

### 6.1 事件循环骨架(run_backtest,backtest.py:73)

```python
for t in range(len(calendar)):          # 逐交易日推进
    # ① 开盘:执行"昨天收盘算出的"信号
    if pending is not _UNSET:  ...先卖后买...
    # ② 收盘:记账 + 用截至今天收盘的数据算评分,生成"明天开盘执行的"信号
    if t + 1 >= cfg.warmup_bars:  ...算分、选股、风控...  pending = frozenset(target_set)
```

**这是反未来函数的物理隔离墙**:信号用 T 日收盘数据(`iloc[:t+1]`),
成交在 T+1 开盘价——模型能看到的信息和它成交的价格之间隔了一个开盘。
有测试(`test_no_lookahead`)锁定:首笔成交必然发生在 warmup 结束之后。

### 6.2 `_UNSET` 哨兵(backtest.py:28)— 三态信号的表达

```python
_UNSET = object()  # 今日无信号(区别于信号为空集 = 目标空仓)
```

引入空仓机制后,"目标持仓"有三态:无信号(不动)/ 目标空仓(清仓)/ 目标某标的(换仓)。
用 `None` 同时表示"无信号"和"空仓"是回测引擎的经典 bug(会导致永远清不了仓),
用单例对象做第三态是最便宜的解法。

### 6.3 组合记账(支持双持仓的重构成果)

```python
positions: dict[str, float] = {}     # symbol -> 股数
cash = cfg.initial_cash
# 开盘执行:先卖出不在目标集合中的 → 再按槽位等权买入空槽
slot = total_equity / cfg.n_hold
```

- `n_hold=1` 时退化为原文的全仓单押(重构后有逐点一致性的验证);
- `n_hold=2` 时每槽 50%,空槽的钱留在 cash(绝对动量过滤把候选过滤光时自然形成半仓现金);
- 卖出回款进 cash,买入从 cash 出——**复利是记账的自然结果**(盈亏自动并入下一份本金),
  对应文章"每次调仓后把之前持仓盈亏相加"。

### 6.4 风控扩展的挂载点(收盘信号段)

```python
# 风控筛选是候选池的过滤器,必须在选股之前:
eligible = candidates.loc[[s for s in candidates.index if _passes_risk_filters(s, data, t, cfg)]]
target_set = select_targets(eligible, set(positions), cfg.rebalance_threshold, cfg.n_hold)
# 入场确认:全空仓时第一名须连续 confirm_days 天居首
# 固定/移动止损 + 止盈(优先级最高):
for sym in list(positions):
    peaks[sym] = max(peaks[sym], close_t)
    if hit_fixed or hit_trailing or hit_take_profit:
        stopped[sym] = t              # 进禁买名单,stop_cooldown_days 天内不买回
        target_set.discard(sym)
```

执行顺序是精心安排的:**先过滤候选池 → 再选股 → 最后止损强平**。

> ⚠️ 第四轮更正:原实现是"先选股 → 再过滤",被剔掉的名额只是**空着**,第 3 名不会递补。
> 因为 V3.1 的溢价规则会把常居第一的 513100 反复剔掉,实测导致 **32.7% 的持仓日只持 1 只**
> (其中 2026-08 连续 22 个交易日),与 `n_hold=2` 的分散意图相悖。现已把筛选移到选股之前,
> 见 `docs/audit-report.md` §9.2 P0-2。

止损优先级最高的原因:它是风险底线,不能被"分数还很高"否决。

三个辅助状态值得注意:

| 状态 | 作用 | 防什么 |
|------|------|--------|
| `peaks[sym]` | 持仓期间最高收盘价 | 移动止损的锚(保护浮盈) |
| `stopped[sym]=t` | 止损禁买名单 | "止损→次日买回→再止损"的追高循环 |
| `top_streak[sym]` | 连续评分第一天数 | 震荡期"一天冠军"式假信号 |

### 6.5 `_passes_risk_filters`(backtest.py:60)— 绝对动量为什么用原始收益

```python
if cfg.min_momentum is not None:
    if close.iloc[-1] / close.iloc[-cfg.slope_n] - 1 <= cfg.min_momentum:
        return False
```

**不能用综合得分(Z-Score)做这个判断**——Z-Score 是横截面相对值,
四只 ETF 全跌时也有一个"相对最高",永远过滤不掉任何东西。
只有原始收益(绝对值)才能表达"全池无趋势"。注释里写了这个道理,是踩坑换来的。

## 七、metrics.py / report.py — 绩效评估与可视化

### 7.1 metrics.py(73 行)

`perf_metrics`(metrics.py:11)从净值序列算出:总收益、年化(CAGR)、年化波动、
夏普、最大回撤、卡玛。公式都是教科书标准(日频 × √252 年化)。

精华在交易配对(改过一次 bug,现在是**按标的、与该 SELL 最近的一笔未平仓 BUY 配对**,即 LIFO):

```python
open_buys: dict = {}
for _, t in trades.sort_values("date").iterrows():
    if t["action"] == "BUY":  open_buys[t["symbol"]] = t
    elif t["action"] == "SELL" and t["symbol"] in open_buys:
        b = open_buys.pop(t["symbol"])   # 同标的最近一笔未平仓 BUY
```

两个教训:① 不能用位置对齐(buys[i]↔sells[i])——截取窗口分析时首笔可能是 SELL,
会配错对、算出 100% 胜率假象;② 双持仓时不同标的买卖交错,必须按标的各自配对。
配不上的"孤儿 SELL"跳过,不造假数据。

### 7.2 report.py(63 行)

`plot_report` 画双联图(对数轴净值 + 回撤填充),叠加四只 ETF 基准。
两个工程细节:`matplotlib.use("Agg")` 无界面后端(服务器可跑);
中文字体回退链 `PingFang SC → Arial Unicode MS`(macOS,Linux 部署需换字体)。

## 八、live.py — 实盘桥(盘前计划与状态持久化)

210 行,对应文章第 4 节的四个实盘机制,**平台无关**(不绑 QMT/Ptrade)。

### 8.1 三段式

```python
plan, new_state = make_plan(data, state, cfg)        # 盘前(8:00):算分→目标→限价单
final = execute_plan(plan, new_state, fills, cfg)    # 开盘(9:30):按成交价执行
store.save_state(final)                              # 收盘后:状态落盘
```

- **`make_plan`**:目标持仓**调用 `backtest.compute_target_set`**——与回测引擎逐行同源的
  唯一实现,不会出现"回测一套、实盘另一套";订单带限价(买=现价×1.01,卖=现价×0.99,
  文章原参数)和**人类可读的换仓理由**;同时返回推进后的 `LiveState`(持仓期峰值锚点、
  禁买名单、评分连击);
- **`execute_plan`**:接收 `fill_prices`(实盘来自券商回报,模拟用开盘价),
  卖出盈亏累加进 `realized_pnl`,回款全额再投(复利)。**非幂等**:同一份 plan 只能执行一次,
  重放会抛错而不是静默改坏账户;
- **`StateStore`**:持仓/风控状态/参数存 JSON(`state/state.json`),日志追加 JSONL。
  测试里专门验证了"新建实例读同一目录 = 程序重启恢复"。

> ⚠️ 历史更正:本文原引用的 `make_plan(data, position, cfg)` / `store.save_position(new_pos)`
> 与 `state/position.json` 是**重写前**的 API 与文件名,均已不存在。
> 另有两个第四轮修掉的实盘缺陷:①`make_plan` 曾丢弃推进后的峰值锚点,
> 使移动止损退化为"锚定开仓价的固定止损";②`confirm_days≥2` 因连击不落盘而永久空仓。
> 详见 `docs/audit-report.md` §9.2 P0-1。

**对接真实券商时只需改一处**:把 `execute_plan` 的 `fill_prices` 换成 QMT/Ptrade 的
成交回报。计划生成、风控、状态管理全部原样复用。

## 九、sweep.py / cli.py — 参数遍历与命令行

### 9.1 sweep.py(47 行)

`weight_grid`(sweep.py:18)在单纯形上生成权重组合(w1+w2+w3=1,步长可调),
`sweep` 做权重×阈值全网格回测,按夏普降序输出。
对应文章"参数是聚宽遍历出来的"——也保留文章的警告:遍历最优是双刃剑。

### 9.2 cli.py(124 行)— 四个子命令

```
fetch      下载/更新数据到 data/
backtest   回测 + 指标 + 图表 + 交易明细(全部参数有 --flag 对应)
sweep      参数遍历 → output/sweep_results.csv
plan       生成今日交易计划(读 state/ 持仓,写日志)
```

`_cfg_from_args`(cli.py:28)把命令行参数通过 `dataclasses.replace` 叠到默认配置上——
新增参数只需往元组里加一个名字,argparse 与配置自动对齐。

## 十、测试体系:102 个测试保什么

```
tests/
├── test_factors.py        因子:手算期望值 + polyfit 独立验证
├── test_scoring.py        Z-Score 性质 + _beats 阈值语义(正/负分区)+ select_targets
├── test_backtest.py       引擎:轮动行为/复利/佣金/_align 交集/confirm_days
├── test_metrics.py        配对回归(窗口截取、孤儿 SELL)+ 佣金扣除 + 夏普口径
├── test_live.py           计划/限价单/持久化/重启恢复/峰值锚点推进/槽位等权/重放保护
├── test_risk_control.py   空仓/过滤/止损/禁买/移动止损/双持仓/溢价规则
├── test_lookahead.py      **确定性无前视**:T 日信号必须与 T+1 及以后行情无关
├── test_config.py         锁住文章披露参数(0.3/0.3/0.4、MOMENTUM_DAY=25、阈值 1.5…)
└── test_sweep.py          权重网格的单纯形约束
```

> **为什么要有 `test_lookahead.py`**:原来的截断实验与未来数据扰动实验**对 off-by-one
> 的窗口偷看完全无区分力**——把 `iloc[:t+1]` 注入成 `iloc[:t+2]` 后,全体测试照样全绿。
> 根因是它们只对"边界那一根 bar"敏感,且旧夹具一律 `open == close`,连"按 T 日收盘成交"
> 这种前视也无人把关。新检验直接断言因果方向,已用变异测试验证有效。

两个测试方法论(值得学):

1. **合成数据锁行为**:不用真实行情,构造"单边上涨 vs 横盘""先暴涨后暴跌"
  这类**行为可预知**的数据,断言引擎做出预期动作——真实数据测试的是"结果对不对",
  合成数据测试的是"逻辑对不对";
2. **单标的池隔离法**:双标的池里 Z-Score 恒有正负,阈值机制总会换仓,
   测不出止损的独立作用;单标的池得分恒 0、基线永不卖出,止损效果才能被单独验证。

## 十一、部署与调用

### 11.1 环境

```bash
cd etf-momentum-rotation
python -m venv .venv && source .venv/bin/activate   # Python ≥ 3.11
pip install -r requirements.txt                      # numpy/pandas/akshare/matplotlib/pytest
# 国内网络加镜像:-i https://pypi.tuna.tsinghua.edu.cn/simple
```

两种运行方式:

```bash
PYTHONPATH=src python -m etf_rotation.cli <command>   # 方式一:免安装
pip install -e . && etf-rotation <command>            # 方式二:装成命令(pyproject.toml 已配)
```

### 11.2 常用命令

```bash
# 数据(首次全量,之后增量;东财限流时加 NO_PROXY 绕过失效的系统代理)
NO_PROXY='*' PYTHONPATH=src python -m etf_rotation.cli fetch

# 回测:原文策略
PYTHONPATH=src python -m etf_rotation.cli backtest

# 回测:推荐改良(V3.1 = 双持仓 + 移动止损 + 动量过滤 + 止盈 + 溢价过滤)
PYTHONPATH=src python -m etf_rotation.cli backtest --preset v3.1
# 等价的手写参数写法:
# PYTHONPATH=src python -m etf_rotation.cli backtest --n-hold 2 --trailing-stop 0.08 \
#     --min-momentum 0 --take-profit 0.25 --premium-max 0.07

# 自定义参数示例
PYTHONPATH=src python -m etf_rotation.cli backtest --slope-n 20 --rebalance-threshold 2.0

# 参数遍历(权重×阈值网格)
PYTHONPATH=src python -m etf_rotation.cli sweep --step 0.1

# 测试
PYTHONPATH=src python -m pytest tests/ -q
```

### 11.3 盘前计划(模拟实盘节奏)

```bash
PYTHONPATH=src python -m etf_rotation.cli plan
```

输出 JSON:当前持仓、目标、四只 ETF 的原始三因子分与综合分、限价订单(含换仓理由),
同时写入 `state/trade_log.jsonl`。可用 crontab 定时(对应文章 8:00 打计划):

```cron
0 8 * * 1-5 cd /path/to/etf-momentum-rotation && NO_PROXY='*' PYTHONPATH=src /path/to/.venv/bin/python -m etf_rotation.cli plan >> state/cron.log 2>&1
```

### 11.4 输出物地图

| 路径 | 内容 |
|------|------|
| `data/*.csv` | 行情缓存(qfq 原名,hfq/腾讯源带后缀) |
| `output/equity_curve.png` | 净值+回撤图 |
| `output/trades.csv` | 全部交易明细 |
| `output/sweep_results.csv` / `drawdown_experiments_r*.csv` | 实验数据 |
| `state/` | 实盘状态:`state.json`(持仓+风控状态)/ `trade_log.jsonl` / `params.json` |

## 十二、设计原则总结(可迁移到其他项目)

1. **单一事实来源**:策略参数集中在一个 frozen dataclass;因子/评分代码回测实盘共用;
2. **默认值即基准**:所有改良做成默认关闭的开关,原版行为随时可复现、可对比;
3. **物理隔离未来函数**:信号(T 收盘)与成交(T+1 开盘)在事件循环结构里强制分离,不靠自觉;
4. **三态用哨兵,不用 None 重载**:`_UNSET` 区分"无信号"与"目标空仓";
5. **宁缺毋假**:数据不足返回 NaN 交给上游,不在底层硬算;
6. **合成数据测逻辑,真实数据测结果**;
7. **小库优先**:sklearn → 十行 numpy;依赖越少,部署越不容易坏。

---

### 推荐阅读顺序

想快速上手:`config.py` → `scoring.py` → `cli.py`(半天)
想吃透策略:`factors.py` → `scoring.py` → `docs/strategy-explained.md`(一天)
想吃透工程:`backtest.py` → `tests/test_backtest.py` → `tests/test_risk_control.py`(一天)
想接实盘:`live.py` → `docs/development.md` 第 7 节(半天)
