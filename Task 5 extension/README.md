# Task 5 extension

固定资源预算下的自行车道安全优化。代码从 Task 5 实验工程提取，保留两个规划任务及其必要输入，可将整个文件夹复制到其他位置运行。

| 命令 | 规划任务 | 预算 |
|---|---|---|
| `full` | 重新选择整个自行车道布局 | 总预算 = `multiplier × C_ref` |
| `incremental` | 保留已有车道，选择新增路段 | `budget` 为绝对新增预算 |

小网络默认采用 Gurobi MILP，大网络默认采用 ALNS。

## 文件结构

```text
Task 5 extension/
├── run.py                         命令行入口、预算设置与结果导出
├── network.py                     TNTP 读取、建设单元归并与路径筛选
├── solver.py                      Gurobi MILP
├── alns.py                        ALNS 搜索
├── requirements.txt
├── README.md
├── data/
│   ├── small/                     小网络：401 节点、177 个正需求 OD
│   └── large/                     扩展网络：1265 节点、711 个正需求 OD
│       # 各含 net.tntp、trips.tntp、n2024_polygon_id_map.csv
└── baselines/
    ├── small/
    └── large/
        # 各含 reference_cost.json 和 fixed_network_j2.json
```

四个 Python 文件构成运行代码。路网文件、节点映射和基准文件随包提供；批量实验调度、历史运行结果和报告生成工具已省去。`network.py`、`solver.py`、`alns.py` 保留原求解实现；`run.py` 的默认输入改为本文件夹内的 `data`。

## 安装

使用 Python 3.10 或更高版本，运行环境为 macOS 或 Linux。ALNS 的路径缓存使用 `fcntl` 文件锁。

在终端进入此文件夹后执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --help
```

Gurobi 求解需要有效许可；仅运行 ALNS 时只需安装 `numpy` 和 `scipy`。依赖的版本范围见 `requirements.txt`。

## 目标函数与服务条件

```text
min J = sum_w q_w [K_w (1 - z_w) + sum_a c_wa]
```

`q_w` 为 OD 需求，`K_w` 为固定的最短路径参考距离，`z_w` 表示是否提供达标路径，`c_wa` 为该路径在路段上的剩余无车道距离。未获得达标路径的需求按 `q_w K_w` 计入负担；达标需求按路径剩余无车道距离计入负担。

默认服务参数为 `--coverage-share 0.8 --epsilon 0.2 --t 0.2`：至少覆盖 80% 的需求，达标路径长度不超过 `1.2 K_w`，无车道长度不超过 `0.2 K_w`。建设变量按物理道路单元计费，两个行驶方向共享车道建设。

软件输出中的 `j2` 字段对应这里的目标函数 `J`，沿用原实验的数据格式。

## 参考成本与已有布局

包内基准直接复用正式实验采用的 Task 3 历史布局与固定布局评价：

| 路网 | `C_ref` | 基准车道单元数 | 基准 `J` |
|---|---:|---:|---:|
| small | 5169.033897138104 | 68 | 446986.9268343237 |
| large | 19469.687353669575 | 242 | 1221979.589822862 |

`reference_cost.json` 保存参考建设成本与布局；`fixed_network_j2.json` 保存同一布局在当前目标下的路径评价。下面的命令同时复用这两个文件，直接进入预算优化。

`C_ref` 是历史可行方案的参考成本。小网络历史成本解的 MIP 间隙约为 0.9553%，大网络历史布局来自 ALNS。

## 完整网络规划

小网络，总预算为 `1.5 C_ref`：

```bash
python run.py full \
  --network small --multiplier 1.5 \
  --reference-cost-result baselines/small/reference_cost.json \
  --baseline-j2-result baselines/small/fixed_network_j2.json \
  --time-limit 18000 --threads 4 \
  --output-dir results/full_small_1p5
```

扩展网络，同样采用 `1.5 C_ref`，使用 ALNS：

```bash
python run.py full \
  --network large --solver alns --multiplier 1.5 \
  --reference-cost-result baselines/large/reference_cost.json \
  --baseline-j2-result baselines/large/fixed_network_j2.json \
  --path-pool-cache results/large_paths.json \
  --time-limit 18000 \
  --output-dir results/full_large_1p5
```

`1.5` 表示总资源上限为参考成本的 150%。正式实验的倍数为 `1.0、1.2、1.5、1.8、2.0`。

若省略 `--reference-cost-result`，小网络入口会尝试重新求解最低建设成本；大网络 ALNS 要求提供参考成本文件。复用上述文件即可沿用本次研究的预算基准。

## 新增自行车道规划

小网络，输入绝对新增预算 `1000`，锁定全部基准车道：

```bash
python run.py incremental \
  --network small \
  --existing-design baselines/small/fixed_network_j2.json \
  --baseline-j2-result baselines/small/fixed_network_j2.json \
  --budget 1000 --time-limit 18000 --threads 4 \
  --output-dir results/incremental_small_1000
```

扩展网络，输入绝对新增预算 `2000`：

```bash
python run.py incremental \
  --network large --solver alns \
  --existing-design baselines/large/fixed_network_j2.json \
  --baseline-j2-result baselines/large/fixed_network_j2.json \
  --budget 2000 --time-limit 18000 \
  --path-pool-cache results/large_paths.json \
  --output-dir results/incremental_large_2000
```

`--budget` 只支付新增设施。已有投入保留且不计入新增预算。正式新增预算档位为 `0、0.05、0.10、0.15、0.20、0.50、0.80、1.00` 倍 `C_ref`；运行时先乘对应的 `C_ref`，再将得到的绝对数值传给 `--budget`。

已有布局可替换为本程序导出的结果 JSON 或建设单元 CSV。若更换布局，应同时提供对应的 `--baseline-j2-result`，或省略该参数让程序重新评价固定布局。

## 常用参数与输出

| 参数 | 含义 |
|---|---|
| `--time-limit` | 最终求解／搜索时限，单位秒；默认 3600，示例采用 18000（5 小时） |
| `--threads` | Gurobi 线程数，默认 4；ALNS 每个案例为单个搜索进程 |
| `--seed` | 随机种子，默认 42 |
| `--path-pool-size` | ALNS 候选路径数量参数，默认 12 |
| `--network-dir PATH` | 使用自定义 TNTP 路网文件夹 |
| `--cost-file FILE.csv` | 完整物理单元成本表，列为 `numeric_u,numeric_v,cost` |
| `--quiet` | 关闭 Gurobi 控制台日志 |

默认建设成本是同一物理单元中有向路段长度的较大值，预算单位随路网长度单位确定。使用货币成本时通过 `--cost-file` 提供完整成本表。更换路网、OD、成本或服务参数后，历史基准文件的实例指纹可能不匹配，需要提供与新实例一致的参考结果。

结果写入指定的 `--output-dir`：

```text
run_summary.json                  预算、算法、基准和最终指标
full_network_j2.json               完整规划结果
incremental_j2.json                新增规划结果
<stage>_chosen_design_units.csv    最终全部车道单元
<stage>_new_design_units.csv       新增规划中新建的车道单元
<stage>_od_metrics.csv             逐 OD 覆盖、距离与目标贡献
<stage>_progress.json              当前阶段与求解进度
```

每次运行按任务产生相应文件。Gurobi 结果记录求解状态、界与间隙；ALNS 结果状态为 `FEASIBLE_HEURISTIC`，间隙为空。退出码 `0` 表示获得经校验的可行结果，`2` 表示输入或参考成本问题，`3` 表示没有获得所需可行结果。

首次检查运行可将示例中的 `--time-limit 18000` 改成 `--time-limit 5`。该参数约束最终优化阶段；读取输入、构建 MILP 和生成 ALNS 路径池也会占用时间。
