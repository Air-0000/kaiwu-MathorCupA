import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import kaiwu


def index(i, j, n):
    return (i - 1) * n + (j - 1)


def check_duplicate(route):
    node_dict = {}
    duplicate = []
    for node in route[1:]:
        if node not in node_dict:
            node_dict[node] = 1
        else:
            node_dict[node] += 1
            duplicate.append(node)
    if duplicate:
        print('duplicate:', duplicate)
    else:
        print('no duplicate')
    return duplicate


def check_missing(route):
    all_lst = set(range(15))
    visited = set()
    for node in route[1:]:
        if node not in visited:
            visited.add(node)
    missing = all_lst - visited
    if missing:
        print('missing:', missing)
    else:
        print('no missing')
    return missing


def check_invalid(route):
    invalid = []
    for node in route[1:]:
        if node > 15 or node < 0:
            invalid.append(node)
    if invalid:
        print('invalid:', invalid)
    else:
        print('no invalid')
    return invalid


def fix_route(route):
    duplicate = check_duplicate(route)
    missing = check_missing(route)
    invalid = check_invalid(route)
    if duplicate or missing or invalid:
        return None
    return route


def time_calculate(route, time_sheet):
    time = 0
    for i in range(len(route) - 1):
        time += time_sheet.loc[route[i], route[i + 1]]
    return time


def three_opt_swap(route, i, j, k):
    route_backup = route.copy()

    A = route_backup[0:i + 1]
    B = route_backup[i + 1:j + 1]
    C = route_backup[j + 1:k + 1]
    D = route_backup[k + 1:]

    route_1 = A + C + B + D
    route_2 = A + B[::-1] + C[::-1] + D
    route_3 = A + C[::-1] + B[::-1] + D
    route_4 = A + B + C[::-1] + D

    unique_routes = []
    for r in [route_1, route_2, route_3, route_4]:
        if r not in unique_routes:
            unique_routes.append(r)
    return unique_routes


def local_search(route, time_sheet, max_iter=50):
    """纯旅行时间局部搜索（3-opt）"""
    curr_route = route.copy()
    curr_time = time_calculate(curr_route, time_sheet)

    iter = 0
    length = len(curr_route)

    print(f"初始: 旅行时间={curr_time:.2f}")

    while iter < max_iter:
        improved = False
        best_route = curr_route.copy()
        best_time = curr_time
        routes_evaluated = 0

        for i in range(1, length - 3):
            for j in range(i + 1, length - 2):
                for k in range(j + 1, length - 1):
                    new_routes = three_opt_swap(curr_route, i, j, k)

                    for r in new_routes:
                        routes_evaluated += 1
                        r_time = time_calculate(r, time_sheet)

                        if r_time < best_time - 1e-6:
                            best_time = r_time
                            best_route = r.copy()
                            improved = True

        print(f'\niter {iter}: 评估了 {routes_evaluated} 条路由, improved={improved}, best_time={best_time:.2f}')

        if improved:
            curr_route = best_route
            curr_time = best_time
            print(f'  -> 新路由: {curr_route}')
            iter += 1
        else:
            print("over: 未找到更优解，退出")
            break

    final_route = curr_route.copy()
    final_time = time_calculate(final_route, time_sheet)
    print(f"最终: 旅行时间={final_time:.2f}")
    return final_route, final_time


def data_loader():
    node_df = pd.read_excel('参考算例.xlsx', sheet_name='节点属性信息')
    time_df = pd.read_excel('参考算例.xlsx', sheet_name='旅行时间矩阵')
    # 用iloc排除第一列（Unnamed: 0），正确提取16x16矩阵（节点0-15）
    travel_time_15_ = time_df.iloc[:16, 1:17]
    # 50节点：排除第一列，取51x51（节点0-50）
    travel_time_50_ = time_df.iloc[:51, 1:52]
    return travel_time_15_, travel_time_50_


def plot_route(route):
    x = np.arange(1, len(route) + 1)
    y = np.array(route, dtype=int)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(x, y, marker="o", linewidth=1.5, color="#4C78A8", label="Hamiltonian")
    ax.set_xlabel("process")
    ax.set_ylabel("node")
    ax.set_title("Route")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False)
    plt.show()
    plt.close(fig)


def solver(context, Q):
    n = context['n']
    M = context['M']
    task = context['task']
    dirname = context['dirname']
    travel_time = context['travel_time']

    Q = (Q + Q.T) / 2
    ising_mat, __ = kaiwu.conversion.qubo_matrix_to_ising_matrix(Q)

    max_val = np.max(np.abs(ising_mat))
    if max_val > 127:
        scale = 127.0 / max_val
        ising_scaled = np.round(ising_mat * scale).astype(np.int8)
    else:
        ising_scaled = ising_mat.astype(np.int8)
        scale = 1.0

    task_name_str = task + str(M)
    checkpoint_dir = './checkpoint/' + dirname
    os.makedirs(checkpoint_dir, exist_ok=True)
    kaiwu.common.CheckpointManager.save_dir = checkpoint_dir

    opti = kaiwu.cim.CIMOptimizer(
        task_name=task_name_str,
        wait=True,
        interval=1,
    )

    solution = opti.solve(ising_scaled)

    Binary = (solution[0][:n * n] + 1) // 2
    X_matrix = Binary.reshape(n, n)

    route = [0]
    for col in range(n):
        for row in range(n):
            if X_matrix[row, col] == 1:
                route.append(row + 1)
    route.append(0)

    print("route", route)
    route_fixed = fix_route(route)
    if not route_fixed:
        return []

    # 局部搜索优化
    final_route, final_time = local_search(route_fixed, travel_time)
    return final_route


def target_time(context, Q):
    n = context['n']
    travel_time = context['travel_time']

    for k in range(1, n):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                a = index(i, k, n)
                b = index(j, k + 1, n)
                Q[a][b] += travel_time.iloc[i - 1, j - 1] / 2

    for i in range(1, n + 1):
        a = index(i, 1, n)
        Q[a][a] += travel_time.iloc[0, i - 1]

    for i in range(1, n + 1):
        a = index(i, n, n)
        Q[a][a] += travel_time.iloc[i - 1, 0]

    return Q


def punish_move(context, Q):
    n = context['n']
    M = context['M']

    for k in range(1, n + 1):
        for i in range(1, n + 1):
            a = index(i, k, n)
            Q[a][a] += -M

        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                a = index(j, k, n)
                b = index(i, k, n)
                Q[a][b] += M

    for i in range(1, n + 1):
        for k in range(1, n + 1):
            a = index(i, k, n)
            Q[a][a] += -M

        for k in range(1, n + 1):
            for l in range(k + 1, n + 1):
                a = index(i, l, n)
                b = index(i, k, n)
                Q[a][b] += M
    return Q


def calculate_result(context):
    N = context['N']
    Q = target_time(context, np.zeros((N, N)))
    Q = punish_move(context, Q)
    return solver(context, Q)


def problem1(travel_time_15, task):
    context = {
        'n': 15,
        'M': 8000,
        'N': 225,
        'task': task,
        'dirname': "problem1",
        'travel_time': travel_time_15
    }
    return calculate_result(context)


if __name__ == '__main__':
    travel_time_15, _ = data_loader()
    route = problem1(travel_time_15, 'problem1-d')
    print("最终路由:", route)
    if route:
        print("总时间:", time_calculate(route, travel_time_15))
