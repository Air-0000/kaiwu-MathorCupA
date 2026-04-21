import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

import kaiwu

def index(i, j, n):
# i : node    |   j : process index (1,1)
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
    """用 .loc 按节点ID索引"""
    time = 0
    for i in range(len(route) - 1):
        time += time_sheet.loc[route[i], route[i + 1]]
    return time

def time_punish_calculate(route, time_sheet, node_df, M1=10, M2=20, S_time=2):
    """
    return : DataFrame with columns [节点, 到达时间, 时间窗下界, 时间窗上界, 惩罚类型, 惩罚值]
    """
    total_penalty = 0
    current_time = 0
    records = []

    for step in range(len(route) - 1):
        from_node = route[step]
        to_node = route[step + 1]

        # 累加旅行时间（用 .loc 按节点ID索引）
        current_time += time_sheet.loc[from_node, to_node]

        # 跳过起点0
        if to_node == 0:
            continue

        # 获取时间窗
        # t_up = 上界 = 更大的值（最晚时间），t_down = 下界 = 更小的值（最早时间）
        t_up = node_df.loc[to_node - 1, '开始服务时间上界']  # 时间窗上限（最晚时间）
        t_down = node_df.loc[to_node - 1, '开始服务时间下界']  # 时间窗下限（最早时间）

        # 计算惩罚
        # 早到：到达时间 < t_down（早于最早时间）
        # 晚到：到达时间 > t_up（晚于最晚时间）
        if current_time < t_down:
            penalty_type = '早到'
            penalty = M1 * (t_down - current_time) ** 2
        elif current_time > t_up:
            penalty_type = '晚到'
            penalty = M2 * (current_time - t_up) ** 2
        else:
            penalty_type = '无'
            penalty = 0

        total_penalty += penalty
        current_time += S_time  # 服务时间

        records.append({
            '节点': to_node,
            '到达时间': current_time - S_time,  # 到达时刻（不含服务时间）
            '时间窗下界': t_down,
            '时间窗上界': t_up,
            '惩罚类型': penalty_type,
            '惩罚值': penalty
        })

    df = pd.DataFrame(records)
    return df, total_penalty

def three_opt_swap(route, i, j, k):
    route_backup = route.copy()

    A = route_backup[0:i + 1]  # 0   --> i
    B = route_backup[i + 1:j + 1]  # i+1 --> j
    C = route_backup[j + 1:k + 1]  # j+1 --> k
    D = route_backup[k + 1:]  # k+1 --> 0

    # route = A + B + C + D
    route_1 = A + C + B + D

    route_2 = A + B[::-1] + C[::-1] + D
    route_3 = A + C[::-1] + B[::-1] + D

    # route_5 = A + B[::-1] + C       + D
    route_4 = A + B + C[::-1] + D

    # 去重：避免不同组合生成相同路线
    unique_routes = []
    for r in [route_1, route_2, route_3, route_4]:
        if r not in unique_routes:
            unique_routes.append(r)
    return unique_routes  # 返回去重后的路线列表

def local_search(route, time_sheet, node_df, max_iter=50, alpha=0.5,mode=1):
    """
    alpha: 旅行时间权重 (1-alpha 是时间窗惩罚权重)
    """
    if mode == 0:
        alpha = 1
    curr_route = route.copy()
    curr_time = time_calculate(curr_route, time_sheet)
    curr_penalty = time_punish_calculate(curr_route, time_sheet, node_df)[1]

    # 归一化：用初始值作为基准，后续改进看相对变化
    init_time = curr_time
    init_penalty = max(curr_penalty, 1e-6)  # 避免除零

    iter = 0
    length = len(curr_route)

    print(f"初始: 旅行时间={curr_time:.2f}, 时间窗惩罚={curr_penalty:.2f}")

    while iter < max_iter:
        improved = False
        best_route = curr_route.copy()
        best_time = curr_time
        best_penalty = curr_penalty
        routes_evaluated = 0

        # 1 <= i < j < k <= len - 2
        for i in range(1, length - 3):
            for j in range(i + 1, length - 2):
                for k in range(j + 1, length - 1):
                    new_routes = three_opt_swap(curr_route, i, j, k)

                    for r in new_routes:
                        routes_evaluated += 1
                        r_time = time_calculate(r, time_sheet)
                        r_penalty = time_punish_calculate(r, time_sheet, node_df)[1]

                        # 归一化目标：alpha * (time/init_time) + (1-alpha) * (penalty/init_penalty)
                        # 这样 time 和 penalty 在相同尺度上比较
                        norm_time = r_time / init_time
                        norm_penalty = r_penalty / init_penalty
                        r_obj = alpha * norm_time + (1 - alpha) * norm_penalty

                        # 当前最优归一化目标
                        norm_curr_time = best_time / init_time
                        norm_curr_penalty = best_penalty / init_penalty
                        best_obj = alpha * norm_curr_time + (1 - alpha) * norm_curr_penalty

                        # 只要有改进就接受（不管多小）
                        if r_obj < best_obj - 1e-8:
                            best_obj = r_obj
                            best_route = r.copy()
                            best_time = r_time
                            best_penalty = r_penalty
                            improved = True

        print(f'\niter {iter}: 评估了 {routes_evaluated} 条路由, improved={improved}')
        print(f'  time={curr_time:.2f}, penalty={curr_penalty:.2f}')

        if improved:
            curr_route = best_route
            curr_time = best_time
            curr_penalty = best_penalty
            print(f'  -> 新路由: {curr_route}')
            iter += 1
        else:
            print("over: 未找到更优解，退出")
            break

    final_route = curr_route.copy()
    final_time = time_calculate(final_route, time_sheet)
    final_penalty = time_punish_calculate(final_route, time_sheet, node_df)[1]
    print(f"最终: 旅行时间={final_time:.2f}, 时间窗惩罚={final_penalty:.2f}")
    return final_route, final_time
    return final_route, final_time

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

def data_loader():
    node_df = pd.read_excel('参考算例.xlsx', sheet_name='节点属性信息')
    time_df = pd.read_excel('参考算例.xlsx',sheet_name='旅行时间矩阵')

    array_time_up = np.array(node_df.loc[:,'开始服务时间上界'])
    array_time_down = np.array(node_df.loc[:,'开始服务时间下界'])
    array_require = np.array(node_df.loc[:,'需求量'])
    combined = np.vstack((array_time_up,array_time_down,array_require))

    travel_time_15_ = time_df.iloc[:16, 1:17]
    travel_time_50_ = time_df.iloc[:51, 1:52]

    return combined,travel_time_15_,travel_time_50_

def build_time_base_distance_matrix(time_features,time_up,alpha=0.7,beta=0.3):
    '''
    :param time_features: 50x50
    :param alpha: time distance weight
    :param beta:  time_up weight
    :return: distance matrix
    '''
    n = len(time_up)

    # 拓展尺度，归一化
    time_up_matrix = np.tile(time_up,(n,1))
    # node i-j的时间上界差异
    time_up_diff_matrix = np.abs(time_up_matrix - time_up_matrix.T)
    scale = MinMaxScaler()
    time_up_diff_norm = scale.fit_transform(time_up_diff_matrix)
    time_dist_norm = scale.fit_transform(time_features)

    combined = alpha*time_dist_norm + beta*time_up_diff_norm
    return combined

def cluster_time_require(time_sheet,require,time_up,mode=0,node_num=50,cluster_num=5):
    '''
    :param mode: 0 时用时间上界（problem3），1时用时间表（problem4）
    :return: List(cluster)
    '''
    # 去掉 0，0这个起点
    node_ids = np.arange(1, node_num + 1)
    time_features = time_sheet.iloc[1:node_num+1,1:node_num+1].values
    if mode == 0:
        combined_dist_matrix = build_time_base_distance_matrix(time_features,time_up[1:])
        option = combined_dist_matrix
    else:
        option = time_features
    hc = AgglomerativeClustering(
        n_clusters=cluster_num,
        metric='precomputed',
        linkage='average',  # ward 只支持 euclidean，改用 average
    )

    labels = hc.fit_predict(option)

    # 期望的require
    total_require = require.sum()
    target_num = total_require / cluster_num

    cluster = [[]for _ in range(cluster_num)]
    # 记录各个聚类的require
    cluster_require = np.zeros(cluster_num)

    for i, (node_id,label) in enumerate(zip(node_ids,labels)):
        temp_cluster = label
        temp_require_deviation = abs(cluster_require[temp_cluster] + require[i] - target_num)
        # c 只是一个序号，不是cluster的什么部分
        for c in range(cluster_num):
            if len(cluster[c]) == 0:
                avg_time_to_cluster = 0
            else:
                cluster_indices = [x - 1 for x in cluster[c]]
                avg_time_to_cluster = time_features[i,cluster_indices].mean()
            # 限定处理范围，避免超出到其他的簇
            if avg_time_to_cluster < 3 :
                new_deviation = abs(cluster_require[c] + require[i] - target_num)
                if new_deviation < temp_require_deviation:
                    temp_cluster = c
                    temp_require_deviation = new_deviation

        cluster[temp_cluster].append(node_id)
        cluster_require[temp_cluster] += require[i]

    # 直接计算和平均的差值，更加直观
    require_sum = [round(d,2) for d in cluster_require]
    average_require = sum(require_sum) / cluster_num
    require_delta = [round(average_require - require_sum[i],2) for i in range(cluster_num)]
    return cluster,require_delta # cluster 应该是（5，10）

# # cluster (cluster_num,cluster_list)
# def select_cluster_center(time_sheet,clusters,time_up):
#     cluster_score = []
#     for cluster_list in clusters:
#         if len(cluster_list) == 0:
#             cluster_score.append(np.inf)
#             continue
#
#         center_time = time_sheet[0,cluster_list]
#         average_center_time = np.mean(center_time)
#
#         # id --> index
#         cluster_indices = [id - 1 for id in cluster_list]
#         # 两次切片，获取簇内元素的 time—sheet
#         intra_cluster_time = time_sheet[cluster_indices,:][:,cluster_indices]
#         avg_intra_time = np.mean(intra_cluster_time)
#         # 中心到达到平均时间 + 簇内平均距离时间
#         approx_arrive_time = average_center_time + avg_intra_time
#
#         loss_list = []
#         for node_id in cluster_list:
#             t = time_up[node_id - 1]
#             if approx_arrive_time > t:
#                 loss = 20 * (t - approx_arrive_time)**2
#             else:
#                 loss = 0
#             loss_list.append(loss)
#
#         average_loss = np.mean(loss_list)
#
#         time_weight = 0.4
#         loss_weight = 0.6
#         total_score = average_loss * loss_weight + average_center_time *time_weight
#
#         cluster_score.append(total_score)
#
#     center_index = np.argmin(cluster_score)
#     center_cluster = clusters[center_index]
#     return center_cluster,center_index

# cluster (cluster_num,cluster_list)
def get_inner_cluster_path(time_sheet,cluster_list):
    if len(cluster_list) == 0:
        return []
    # 直接用 .loc 按节点ID取子矩阵
    intra_dist_sub_matrix = time_sheet.loc[cluster_list, cluster_list].values

    current_index = 0 # 起点
    visited_nodes = [False] * len(cluster_list)
    visited_nodes[current_index] = True #起点已经去过
    # 初始化内部路径的时候直接加入起点
    inner_path = [cluster_list[current_index]]

    # 只要inner path没有遍历完，就不停下来
    while len(inner_path) < len(cluster_list):
        # intra_dist_sub_matrix （3x3） --> current_dist (1x3)
        current_dist = intra_dist_sub_matrix[current_index]
        # d or inf , 迫使选择没有去过的路径点 i: index , d: distance
        min_dist_index = np.argmin([d if not visited_nodes[i] else np.inf for i, d in enumerate(current_dist)])
        # 添加路径点，更换current_index成新的，min_dist_index一定是新的
        visited_nodes[min_dist_index] = True
        inner_path.append(cluster_list[min_dist_index])
        current_index = min_dist_index

    return inner_path

def optimize_inner_cluster_path(time_sheet,cluster_list,task,combined):
    cluster_len = len(cluster_list)
    if cluster_len <= 2:
        return cluster_list
    # 用节点ID作为标签提取子矩阵
    center_with_labels = [0] + cluster_list
    intra_time_df = time_sheet.loc[center_with_labels, center_with_labels].copy()

    context = {
        'M1': 10,
        'M2': 20,
        'n': cluster_len,  # 簇内客户数（不含0）
        'M': 8000,   # 惩罚系数，复用problem1的经验值
        'N': cluster_len*cluster_len,  # Q矩阵维度：n个客户×n个步骤
        'S_time': 2,
        'task': task,
        'M_time': 100,
        'combined': combined,
        'dirname': f"problem3_4",
        'travel_time': intra_time_df.iloc[1:, 1:],  # QUBO用：位置索引（0到n-1）
        'travel_time_node': intra_time_df,  # 路由评估用：节点ID索引
        'cluster_list': cluster_list  # QUBO行索引到实际节点ID的映射
    }
    quantum_route = calculate_result(context)

    #  提取簇内路径
    if not quantum_route or len(quantum_route) < 3:
        return cluster_list
    intra_quantum_path = [node for node in quantum_route if node != 0]

    # 是否 missing
    if set(intra_quantum_path) != set(cluster_list):
        return cluster_list
    return intra_quantum_path

def get_optimized_path(time_sheet,task,combined,mode=0):
    time_up = combined[0]
    require = combined[2]
    clusters, require_delta = cluster_time_require(time_sheet,require,time_up,mode)
    print("聚类结果（每类客户ID）：", clusters)
    print("每类需求偏差：", require_delta)
    '''
    all_inner_paths = []
    for i, cls in enumerate(clusters):
        inner_path = get_inner_cluster_path(travel_time_50,cls)
        all_inner_paths.append(inner_path)

    # 同时获取 （一条路线 ， 五条路线）
    only_path = [0]
    all_global_paths = []
    for path in all_inner_paths:
        only_path.append(path)

        global_path = [0]
        global_path.append(path)
        global_path.append(0)
        all_global_paths.append(global_path)

    only_path.append(0) # 添加结尾
    '''

    all_inner_paths_opti = []
    for i, cls in enumerate(clusters):
        print(f"\n优化第{i + 1}簇（客户数：{len(cls)}）...")
        quantum_inner = optimize_inner_cluster_path(time_sheet,cls,task,combined)
        vehicle_inner = [0] + quantum_inner + [0]
        vehicle_inner_fixed = fix_route(vehicle_inner)
        if not vehicle_inner_fixed:
            vehicle_inner_fixed = [0] + get_inner_cluster_path(time_sheet,cls) + [0]
        all_inner_paths_opti.append(vehicle_inner_fixed)
        print(f"第{i + 1}簇初始路径：{vehicle_inner_fixed} | 时间：{time_calculate(vehicle_inner_fixed, time_sheet):.2f}")

    only_path = [0]
    for path in all_inner_paths_opti:
        only_path.extend([node for node in path if node != 0])  # 去掉路径中的0，避免重复
    only_path.append(0)
    return all_inner_paths_opti,only_path

def solver(context,Q):
    '''
    :return: route
    '''
    n = context['n']
    M = context['M']
    task = context['task']
    dirname = context['dirname']
    travel_time = context['travel_time']
    # qubo  -->  ising (matrix)
    Q = (Q + Q.T) / 2  # symmetrize first
    ising_mat, __ = kaiwu.conversion.qubo_matrix_to_ising_matrix(Q)

    # 8-bit 有符号范围: -128 到 127
    max_val = np.max(np.abs(ising_mat))

    # 缩放系数到 8-bit 范围
    if max_val > 127:
        scale = 127.0 / max_val
        ising_scaled = np.round(ising_mat * scale).astype(np.int8)
    else:
        ising_scaled = ising_mat.astype(np.int8)
        scale = 1.0

    task_name_str = task + str(M)

    checkpoint_dir = './kaiwu/checkpoint/'+ dirname
    os.makedirs(checkpoint_dir, exist_ok=True)

    kaiwu.common.CheckpointManager.save_dir = checkpoint_dir

    opti = kaiwu.cim.CIMOptimizer(
        task_name=task_name_str,
        wait=True,
        interval=1,
    )

    solution = opti.solve(ising_scaled)

    # ising {-1,1}  ---->  binary {0,1}
    Binary = (solution[0][:n*n] + 1) // 2

    X_matrix = Binary.reshape(n, n)

    '''  X Matrix   reshape之后（one hot）

                步骤1  步骤2  ...  步骤15
        节点1     1      0     ...    0
        节点2     0      1     ...    0
        ...
        节点15    0      0     ...    1
    '''

    # 从结果解析线路 列是步骤
    route = [0]

    # cluster_list: QUBO行索引到实际节点ID的映射
    cluster_list = context.get('cluster_list', None)

    for col in range(n):
        for row in range(n):
            if X_matrix[row, col] == 1:
                if cluster_list:
                    route.append(cluster_list[row])  # 用cluster_list映射回实际节点ID
                else:
                    route.append(row + 1)  # 默认：位置+1作为节点ID

    route.append(0)  # 结尾回到 0

    print("route", route)
    route_fixed = fix_route(route)

    if not route_fixed:
        return []

    # 加载节点属性用于时间窗惩罚计算
    node_df = pd.read_excel('参考算例.xlsx', sheet_name='节点属性信息')
    # travel_time_node 是 intra_time_df，用节点ID索引
    travel_time_node = context.get('travel_time_node', travel_time)

    # local_search 现在用 .loc 索引，route 和 travel_time_node 都用节点ID
    final_route, final_time = local_search(route_fixed, travel_time_node, node_df,1)
    return final_route

def target_time(context, Q):
    n = context['n']
    travel_time = context['travel_time']
    # add time in to matrix

    for k in range(1,n):
        for i in range(1,n+1):
            for j in range(1,n+1):
                a = index(i,k,n)
                b = index(j,k+1,n)

                # a, b equal to i, j in Q_index
                Q[a][b] += travel_time.iloc[i-1,j-1] / 2

    for i in range(1,n+1):
        a = index(i,1,n)
        Q[a][a] += travel_time.iloc[0,i-1]

    for i in range(1,n+1):
        a = index(i,n,n)
        Q[a][a] += travel_time.iloc[i-1,0]

    return Q

def punish_move(context,Q):
    n = context['n']
    M = context['M']

    # add M(..) to matrix

    # M Sigma_i( Sigma_j(x_ij) - 1)^2    一次去一个
    for k in range(1,n+1):
        # 对角线
        for i in range(1,n+1):
            a = index(i,k,n)
            b = index(i,k,n)
            Q[a][b] += -M

        # 非对角线 遍历每一个k步去了j或i的可能
        for i in range(1,n+1):
            for j in range(i+1,n+1):
                a = index(j,k,n)
                b = index(i,k,n)

                Q[a][b] += M

    # M Sigma_j( Sigma_i(x_ij) - 1)^2   一个只能去一次 （i，j对掉）
    for i in range(1,n+1):
        # 对角线
        for k in range(1,n+1):
            a = index(i,k,n)
            b = index(i,k,n)
            Q[a][b] += -M

        # 同理遍历了i节点时是j，k的可能
        for k in range(1,n+1):
            for l in range(k+1,n+1):
                a = index(i,l,n)
                b = index(i,k,n)

                Q[a][b] += M
    return Q

def punish_time(context,Q):
    n = context['n']
    M1 = context['M1']
    M2 = context['M2']
    travel_time = context['travel_time']
    S_time = context['S_time']
    combined=context['combined']

    node_time_up = combined[0]
    node_time_down = combined[1]

    # 估计每步平均旅行时间
    valid_times = travel_time[travel_time > 0]
    avg_step_time = np.mean(valid_times)

    # 综合估算参数
    avg_from_problem1 = 29 / 15

    for k in range(1, n + 1):

        for i in range(1, n + 1):
            # 综合估算
            min_to_i = min(travel_time.iloc[0, i-1], travel_time.iloc[i-1, 0])

            # 加权平均
            estimated_arrival = k * (0.5 * avg_from_problem1 + 0.5 * min_to_i)

            a = index(i, k, n)

            t_up = node_time_up[i-1]
            t_down = node_time_down[i-1]

            # 早到惩罚 (arriving before earliest time t_down)
            if estimated_arrival < t_down:
                early = t_down - estimated_arrival
                penalty = M1 * early * early
                Q[a, a] += penalty

            # 晚到惩罚 (arriving after latest time t_up)
            if estimated_arrival > t_up:
                late = estimated_arrival - t_up
                penalty = M2 * late * late
                Q[a, a] += penalty

    return Q


def calculate_result(context, mode=1):
    N = context['N']

    #   Q = t + M(..) + M(..) + ..
    Q = target_time(context, np.zeros((N, N)))
    Q = punish_move(context, Q)

    if mode == 1:
        Q = punish_time(context, Q)
    return solver(context, Q)


def problem1(travel_time_15,task,combined=None):
    '''
    #     题目1: 单车辆，最小时间，无特殊时间限制
    #
    #     i : 当前node
    #     j : 下一node
    #     k / l : 步骤
    #
    #   return : route
    # '''
    # the best : 'problem1-d8000' -- 29
    # [0, 13, 10, 8, 1, 3, 6, 7, 14, 9, 12, 5, 2, 15, 4, 11, 0]
    context = {
        'n': 15,
        'M': 8000,
        'N': 225,
        'task': task,
        'dirname':"problem1",
        'travel_time': travel_time_15
    }

    # 'problem1-d8000' -- 29
    # 'problem1-t8000' -- 30
    # 'problem1-s8000' -- 30
    # 'problem1-z8000' -- 30
    # 'problem1-v8000' -- 31
    # 'problem1-w8000' -- 31

    return calculate_result(context,0)

def problem2(travel_time_15,task,combined):
    '''
        题目2: 单车辆，最小时间，有时间窗，服务时间限制
        时间惩罚用 HOBO 建模
    '''
    context = {
        'n':15,
        'M':8000,
        'N':225,
        'M1':10,
        'M2':20,
        'S_time':2,
        'task':task,
        'combined':combined,
        'dirname':"problem2",
        'travel_time':travel_time_15
    }

    return calculate_result(context, 1)


def problem3(travel_time_50,task,combined):
    five_route, one_route = get_optimized_path(travel_time_50,task,combined,0)
    return one_route

def problem4(travel_time_50,task,combined):
    five_route, one_route = get_optimized_path(travel_time_50,task,combined,1)
    return five_route

def test(route, travel_time, node_df):
    total_time = 0
    time_details = []

    # 遍历每段路程，从Excel取时间并累加
    for i in range(len(route) - 1):
        start = int(route[i])
        end = int(route[i + 1])
        # 用 .iloc 按位置索引，避免标签重复问题
        segment_time = travel_time.iloc[start, end]
        total_time += segment_time
        time_details.append(f"{start}→{end}: {segment_time}分钟")

    # 计算时间窗惩罚
    penalty_df, total_penalty = time_punish_calculate(route, travel_time, node_df)

    # 打印结果
    print("每段路程时间明细：")
    print(" | ".join(time_details))
    print(f"\n自动计算的总时间：{total_time}分钟")
    print(f"\n时间窗惩罚总惩罚值：{total_penalty:.2f}")
    print("\n各节点时间窗惩罚明细：")
    print(penalty_df.to_string(index=False))

    return penalty_df, total_penalty

if __name__ == '__main__':

    combined,travel_time_15,travel_time_50 = data_loader()
    node_df = pd.read_excel('参考算例.xlsx', sheet_name='节点属性信息')
    test(problem1(travel_time_50,'problem1-d',combined),travel_time_50, node_df)

