from pathlib import Path
import matplotlib.pyplot as plt
import json

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_vehicle_count_compare(results, output):
    labels = [str(item["vehicle_limit"]) for item in results]
    objectives = [item["final_solution"]["objective"] for item in results]
    used = [item["final_solution"]["used_vehicle_count"] for item in results]
    travel = [item["final_solution"]["travel_time"] for item in results]
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    bars = ax.bar(range(len(labels)), objectives, color="#4C78A8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("可用车辆数量")
    ax.set_ylabel("旅行时间 + 时间窗口惩罚")
    ax.set_title("Q4 不同车辆限制下的目标")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    for bar, obj, u, tr in zip(bars, objectives, used, travel, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 800,
            f"使用数量={u}\n时间={tr}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=280, bbox_inches="tight")
    plt.close(fig)


def plot_route_sequences(vehicle_results, output):
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    colors = ["#4C78A8", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7"]
    for idx, route_result in enumerate(vehicle_results):
        route = route_result["route"]
        if len(route) <= 2:
            continue
        y = [idx + 1] * len(route)
        ax.plot(range(len(route)), y, marker="o", linewidth=1.5, color=colors[idx % len(colors)], label=f"车辆 {idx+1}")
        for x, node in enumerate(route):
            ax.text(x, idx + 1 + 0.06, str(node), ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("访问每条车辆路线的位置")
    ax.set_ylabel("车辆序号")
    ax.set_title("Q4 最终多车路线序列")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=280, bbox_inches="tight")
    plt.close(fig)


def plot_vehicle_loads(vehicle_results, capacity, output):
    labels = [f"V{item['vehicle_id']}" for item in vehicle_results]
    loads = [item["load"] for item in vehicle_results]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.bar(labels, loads, color="#59A14F")
    ax.axhline(capacity, color="#E15759", linestyle="--", linewidth=1.4, label="承载量")
    ax.set_ylabel("车辆负载")
    ax.set_title("Q4 最终车辆负载")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False)
    for bar, val in zip(bars, loads, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.6, str(val), ha="center", va="bottom", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=280, bbox_inches="tight")
    plt.close(fig)

# ===================== 核心：读取外部JSON文件 + 绘图 =====================
if __name__ == '__main__':
    # 1. 配置文件路径
    json_file_path = Path("output_json/q4_result.json")  # JSON文件名，和代码同目录
    output_dir = Path("output")             # 图表输出文件夹

    # 2. 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f" 成功读取数据文件：{json_file_path}")
    except FileNotFoundError:
        print(f" 错误：未找到文件 {json_file_path}，请检查文件是否存在！")
        exit()
    except Exception as e:
        print(f" 读取文件失败：{str(e)}")
        exit()

    vehicle_compare_raw = data["vehicle_count_comparison"]
    plot_results = []
    for limit, res in vehicle_compare_raw.items():
        plot_results.append({
            "vehicle_limit": int(limit),
            "final_solution": res
        })
    # 车辆详情 + 容量
    vehicle_details = data["final_answer"]["vehicle_details"]
    vehicle_capacity = data["scope"]["vehicle_capacity"]

    # 4. 生成图表
    print("\n生成图1: 不同车辆限制目标值对比...")
    plot_vehicle_count_compare(plot_results, output_dir / "q4_vehicle_count_compare.png")

    print("\n生成图2: 多车辆路线序列图...")
    plot_route_sequences(vehicle_details, output_dir / "q4_route_sequences.png")

    print("\n生成图3: 车辆负载分布图...")
    plot_vehicle_loads(vehicle_details, vehicle_capacity, output_dir / "q4_vehicle_loads.png")

    # 完成提示
    print("\n" + "="*60)
    print(" 所有图表生成完成！已保存至 output 文件夹")
    print("="*60)