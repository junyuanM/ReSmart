import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from tqdm import *
from sklearn.preprocessing import normalize
# file_path = './dataset/metro-trips-2024-q1/new_data.csv'
file_path = './dataset/food/new_data.csv'
# file_path = './dataset/driver/new_data.csv'
# file_path = './dataset/yellow/yellow_new_data.csv'
num_players = 20
num_arms = 20
num_time_slots = 100000
L = 50000000
arm_to_player_preferences = [random.sample(range(num_players), num_players) for _ in range(num_arms)]
#print("arm_to_player_preference:", arm_to_player_preferences)
choosing_result = [[-1] * num_players for _ in range(num_time_slots)]  # 用于记录每个时间步的选择结果
matching_result = [[-1] * num_players for _ in range(num_time_slots)]  # 初始化matching_result矩阵
alpha = [[1] * num_arms for _ in range(num_players)]
beta = [[1] * num_arms for _ in range(num_players)]
theta = [[0] * num_arms for _ in range(num_players)]
feasible_set = [[] for _ in range(num_players)]
data = 0
Y = [[0] * num_arms for _ in range(num_players)]
candidate_arm = [[] for _ in range(num_arms)]
r = [[0] * num_arms for _ in range(num_players)]
data_r = [[0] * num_arms for _ in range(num_players)]
#regrets = [[] for _ in range(num_players)]
regrets = [[0] * num_time_slots for _ in range(num_players)]
utilities = [[0] * num_time_slots for _ in range(num_players)]
conflict_times = [0] * num_time_slots
conflict_sum = []
def update_conflict(time_slot):
    global conflict_times
    for j in range(num_arms):
        # if not candidate_arm[j]:
        #     conflict_times[time_slot] += 1
        if len(candidate_arm[j]) >= 2:
            conflict_times[time_slot] += 1
    #conflict_times[time_slot] += num_players - num_arms

def update_feasible_set(time_slot):
        global feasible_set
        feasible_set = [[] for _ in range(num_players)]
        #将当前匹配的每个arm的最大preference所对应的player_id输出成一维矩阵
        current_max_arm_pre = [-1] * num_arms # 初始化为-1
        for i in range(num_players):
            mi = matching_result[time_slot -1][i]
            if mi == -1:#if no match for player i
                continue
            current_max_arm_pre[mi] = i
        #start to update feasible set
        for j in range(num_arms):
            if current_max_arm_pre[j] != -1:
                k = 0
                while arm_to_player_preferences[j][k] != current_max_arm_pre[j]:
                    #print(arms[j].player_preference[k])
                    feasible_set[arm_to_player_preferences[j][k]].append(j)
                    k += 1
                feasible_set[arm_to_player_preferences[j][k]].append(j)
            #########################################将没有匹配的arm加入feasible_set##########################################################
            else:# if no match for arm j
                for l in range(num_players):
                    feasible_set[l].append(j)

def update_r():
    global data_r
    global r
    for i in range(num_players):
        data_r[i] += np.random.normal(scale = 0.1, size = num_arms)
    max = np.max(data_r)
    min = np.min(data_r)
    for i in range(num_players):
        for j in range(num_arms):
            data_r[i][j] = (max - data_r[i][j])/(max - min)
    for i in range(num_players):
        temp = sorted(data_r[i])
        t_r = [temp.index(data_r[i][j]) for j in range(num_arms)]
        r[i] = [t_r[j]/num_arms for j in range(num_arms)]

def run_market():
        global alpha
        global beta
        global choosing_result
        global matching_result
        global Y
        global theta
        global candidate_arm
        global regrets
        global utilities
        global conflict_sum
        k = 1
        sum = 0
        for time_slot in tqdm(range(1, num_time_slots)):
            #print("time_slot:", time_slot)
            #start algorithm
            if k == 1:
                #Set 𝑛𝑡𝑖,𝑗 = 0, 𝛼𝑖,𝑗 = 1, 𝛽𝑖,𝑗 = 1, ∀𝑖 ∈ T, ∀𝑗 ∈ U
                alpha = [[1] * num_arms for _ in range(num_players)]
                beta = [[1] * num_arms for _ in range(num_players)]
                #∀𝑖 ∈ T, 𝑎1 (𝑖) = 𝑗, 𝑗 ∼ U uniformly at random.
                update_r()
                candidate_arm = [[] for _ in range(num_arms)]
                for i in range(num_players):
                    choosing_result[time_slot - 1][i] = random.randint(0, num_arms - 1)
                # for i in range(num_players):
                #     matching_result[time_slot -1][i] = choosing_result[time_slot - 1][i]
                for i in range(num_players):
                    matching_result[time_slot -1][i] = -1
                    candidate_arm[choosing_result[time_slot - 1][i]].append(i)
                for j in range(num_arms):
                    player_index = []
                    if not candidate_arm[j]:
                        continue
                    for p_i in candidate_arm[j]:
                        player_index.append(arm_to_player_preferences[j].index(p_i))
                    Min = min(player_index)
                    minindex = player_index.index(Min)
                    matching_result[time_slot -1][candidate_arm[j][minindex]] = j
                # print("choosing_result:", choosing_result)
            #Distributed Task Matching using ThompsonSampling (DTTS) start.
            # print(matching_result)
            #Update the feasible set
            update_feasible_set(time_slot)
            # print("feasible_set:", feasible_set)
            Y = [[0] * num_arms for _ in range(num_players)]
            candidate_arm = [[] for _ in range(num_arms)]
            for i in range(num_players):
                #𝜃𝑖,𝑗 ∼ 𝐵𝑒 (𝛼𝑖,𝑗, 𝛽𝑖,𝑗)
                for j in range(num_arms):
                    theta[i][j] = np.random.beta(alpha[i][j], beta[i][j])
                    #theta[i][j] = np.random.normal(loc=r[i][j], scale=10, size=None)
                #Draw 𝐵𝑖(𝑡) ∼ 𝐵𝑒𝑟(𝜆) independently.
                lambda_prob = 0.1    #lambda
                if random.random() < lambda_prob:#lambda
                    #set at(i) = at-1(i)
                    choosing_result[time_slot][i] = choosing_result[time_slot - 1][i]
                else:
                    #updating_feasible_set is in initialization
                    #Attempt to propose the matching request to 𝑎 ∈ 𝐹𝑡 (𝑖) with the maximum 𝜃𝑖,𝑗 .
                    if not feasible_set[i]:
                        choosing_result[time_slot][i] = choosing_result[time_slot - 1][i]
                    else:
                        a = feasible_set[i][0]
                        for arm in feasible_set[i]:
                            if theta[i][arm] > theta[i][a]:
                                a = arm
                        #Set 𝑎𝑡 (𝑖) = 𝑎
                        choosing_result[time_slot][i] = a
                candidate_arm[choosing_result[time_slot][i]].append(i)
                #if 𝜋¯𝑎𝑡 (𝑖)(𝑖) ≻𝑎𝑡 (𝑖) 𝜋¯𝑖′ (𝑖),𝑖′ ∈ T𝑡𝑖,j
            # print("choosing_result:", choosing_result)
            # print("candidate_arm:", candidate_arm)
            for i in range(num_players):
                matching = choosing_result[time_slot][i]
                player_index = []
                # for k in candidate_arm[matching]:
                #     for p in range(num_players):
                #         if arm_to_player_preferences[matching][p] == k:
                #             player_index.append(p)
                for k in candidate_arm[matching]:
                    player_index.append(arm_to_player_preferences[matching].index(k))
                Min = min(player_index)
                minindex = player_index.index(Min)
                # print("i:", i)
                # print("matching:", matching)
                # print("minindex:", minindex)
                if candidate_arm[matching][minindex] == i:
                    #print(time_slot)
                    #Obtain a utility 𝑋t(i, mt(i))，正态分布
                    #X[i][a] = np.random.noraml(r[i][a], 1)########正太分布大于1不能后续用伯努利
                    #draw 𝑌𝑡(i, mt(i))
                    #print(X[i][a])
                    #𝑚𝑡 (𝑖) = 𝑎𝑡 (𝑖).
                    # print(type(matching))
                    # print(type(choosing_result))
                    # print(time_slot, i)
                    matching_result[time_slot][i] = choosing_result[time_slot][i]
                    if random.random() < r[i][matching]:
                        Y[i][matching] = 1
                    else:
                        Y[i][matching] = 0
                    #Update parameter of Beta distribution 𝐵𝑒:
                    alpha[i][matching] += Y[i][matching]
                    beta[i][matching] = beta[i][matching] + 1 - Y[i][matching]
                    #update regret
                    regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i]) - Y[i][matching]
                    utilities[i][time_slot] = utilities[i][time_slot - 1] + Y[i][matching]
                else:
                    regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i])
                    utilities[i][time_slot] = utilities[i][time_slot - 1]
                    #regrets[i][time_slot] = min(r[i]) - Y[i][matching_result]
            # print("alpha:", alpha)
            # print("beta:", beta)
            # print("utilities:", utilities)
            # print("matching_result:", matching_result)
            #计算冲突
            update_conflict(time_slot)
            sum += conflict_times[time_slot]
            if time_slot % 5000 == 0:
                conflict_sum.append(sum/5000)
                sum = 0
            if k <= L:########L???
                k += 1
            else:
                k = 1

#get r
def gen_r(file_path):
    # 读取csv文件
    global r
    global data_r
    df = pd.read_csv(file_path)
    data = df.iloc[1:,].values.transpose()
    num_columns = data.shape[1]
    selected_columns = np.random.choice(num_columns, size=num_arms, replace=False)
    data = data[:,selected_columns]
    data[np.isnan(data)] = 0
    theta = np.random.rand(num_players, data.shape[0])
    data_r = np.dot(theta, data)
    #r = normalize(r, norm = 'l2', axis = 1)
    max = np.max(data_r)
    min = np.min(data_r)
    for i in range(num_players):
        for j in range(num_arms):
            data_r[i][j] = (max - data_r[i][j])/(max - min)
    for i in range(num_players):
        temp = sorted(data_r[i])
        t_r = [temp.index(data_r[i][j]) for j in range(num_arms)]
        r[i] = [t_r[j]/num_arms for j in range(num_arms)]
    return r

def plot_chart(chart_type, x_values, y_values, title, xlabel, ylabel, color):
    if chart_type == 'bar':
        # 创建柱状图
        plt.bar(x_values, y_values, color=color)
    elif chart_type == 'line':
        # 创建折线图
        plt.plot(x_values, y_values, color=color, marker='o', linestyle='-', linewidth=2)

    # 添加标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    # 显示图形
    plt.show()

def main():
    # array = [1, 2, 3]
    # arr = [1, 2]
    
    # print(array == arr)

    # hhh
    # 设置参数
    r = gen_r(file_path)
    run_market()
    print(regrets)
    time_slots = range(1, num_time_slots + 1)  # 时间槽
    time_thousand = range(100)
    av_regret = [0] * num_time_slots
    for j in range(num_time_slots):
        for i in range(num_players):
            av_regret[j] += regrets[i][j]
        av_regret[j]  = av_regret[j] / num_players
    #av_utility = np.sum(market.utilities, axis=0)/num_players
    av_utility = [0] * num_time_slots
    for j in range(num_time_slots):
        for i in range(num_players):
            av_utility[j] += utilities[i][j]
        av_utility[j]  = av_utility[j] / num_players
    print(utilities)
    print("alpha:", alpha)
    print("beta:", beta)
    for l in range(10):
        print(matching_result[num_time_slots - 10 + l])
    stable_result = []
    s = 0
    for i in range(1, num_time_slots):
        if matching_result[i] != matching_result[i - 1]:
            s += 1
        if i % 100 == 0:
            stable_result.append(s)
            s = 0
    # for i in range(1, num_time_slots):
    #     if matching_result[i] != matching_result[i - 1]:
    #         s += 1
    #     stable_result.append(s)
    # plot_chart('line', time_slots, av_regret, 'regret', 'time', 'regret', 'blue')
    # plot_chart('line', time_slots, av_utility, 'utility', 'time', 'utility', 'blue')
    plot_chart('line', time_thousand, av_regret[:100], 'regret', 'time', 'regret', 'blue')
    plot_chart('bar', time_thousand, av_utility[:100], 'utility', 'time', 'utility', 'blue')
    plot_chart('line', time_slots, conflict_times, 'conflict', 'time', 'conflict_times', 'blue')
    time_sum = [i for i in range(int(num_time_slots/500) - 1)]
    plot_chart('line', time_sum, conflict_sum, 'conflict_sum', 'time', 'conflict_sum', 'blue')
    plot_chart('line', [i for i in range(int(num_time_slots/100) - 1)], stable_result, 'stable_result', 'time', 'stable_result', 'blue')
    #plot_chart('line', time_slots, stable_result, 'stable', 'time', 'stable', 'blue')
main()