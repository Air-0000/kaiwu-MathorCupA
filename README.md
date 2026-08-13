# kaiwu-MathorCupA 🧮

MathorCup 数学建模竞赛 A 题：运用量子计算解决路径规划的组合优化问题。

## 赛题

MathorCup A 题 —— 组合优化路径规划，探索量子计算方法（如 QAOA、量子退火）在路径规划问题上的应用。

## 文件结构

| 文件 | 说明 |
|------|------|
| `q1.py` ~ `q4.py` | 四个问题的求解代码 |
| `main.py` | 主入口 |
| `png/` | 各问的可视化结果（最优路径、车辆数对比、违反约束统计等） |
| `AMC260计算结果.xlsx` | 计算结果表 |
| `AMC260附件.zip` | 赛题附件数据 |
| `AMC260承诺书.jpg` | 参赛承诺书 |

## 可视化

`png/` 目录包含 Q1-Q4 的最优路线图、种子对比、违规统计、车辆负载对比等图表，由 `png/q1_2_3_visual.py`、`png/q4_visual.py` 生成。

## 依赖

```bash
pip install numpy matplotlib pulp  # 及量子计算相关库（按实际使用）
```
