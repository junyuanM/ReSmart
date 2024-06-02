import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import normalize

# 全局变量设置
file_path = './dataset/food/new_data.csv'
num_players = 20
num_arms = 20
r = [[0] * num_arms for _ in range(num_players)]
num_time_slots = 100000
L = 50000000
INF = float('inf')
arm_to_player_preferences = [random.sample(range(num_players), num_players) for _ in range(num_arms)]

# 函数定义
def update_conflict(time_slot, candidate_arm, conflict_times):
    for j in range(len(candidate_arm)):
        if len(candidate_arm[j]) >= 2:
            conflict_times[time_slot] += 1

def update_feasible_set(time_slot, num_players, num_arms, arm_to_player_preferences, matching_result, feasible_set):
    feasible_set = [[] for _ in range(num_players)]
    current_max_arm_pre = [-1] * num_arms
    for i in range(num_players):
        mi = matching_result[time_slot -1][i]
        if mi == -1:
            continue
        current_max_arm_pre[mi] = i
    for j in range(num_arms):
        if current_max_arm_pre[j] != -1:
            k = 0
            while arm_to_player_preferences[j][k] != current_max_arm_pre[j]:
                feasible_set[arm_to_player_preferences[j][k]].append(j)
                k += 1
            feasible_set[arm_to_player_preferences[j][k]].append(j)
        else:
            for l in range(num_players):
                feasible_set[l].append(j)
    return feasible_set


def run_market_thompson_sampling():
    choosing_result = [[-1] * num_players for _ in range(num_time_slots)]
    matching_result = [[-1] * num_players for _ in range(num_time_slots)]
    alpha = [[1] * num_arms for _ in range(num_players)]
    beta = [[1] * num_arms for _ in range(num_players)]
    theta = [[0] * num_arms for _ in range(num_players)]
    feasible_set = [[] for _ in range(num_players)]
    candidate_arm = [[] for _ in range(num_arms)]
    regrets = [[0] * num_time_slots for _ in range(num_players)]
    utilities = [[0] * num_time_slots for _ in range(num_players)]
    conflict_times = [0] * num_time_slots
    conflict_sum = []

    k = 1
    sum_conflicts = 0
    for time_slot in tqdm(range(1, num_time_slots)):
        if k == 1:
            alpha = [[1] * num_arms for _ in range(num_players)]
            beta = [[1] * num_arms for _ in range(num_players)]
            candidate_arm = [[] for _ in range(num_arms)]
            for i in range(num_players):
                choosing_result[time_slot - 1][i] = random.randint(0, num_arms - 1)
            for i in range(num_players):
                matching_result[time_slot -1][i] = -1
                candidate_arm[choosing_result[time_slot - 1][i]].append(i)
            for j in range(num_arms):
                if candidate_arm[j]:
                    player_index = [arm_to_player_preferences[j].index(p) for p in candidate_arm[j]]
                    Min = min(player_index)
                    minindex = player_index.index(Min)
                    matching_result[time_slot -1][candidate_arm[j][minindex]] = j
        feasible_set = update_feasible_set(time_slot, num_players, num_arms, arm_to_player_preferences, matching_result, feasible_set)
        theta = [[np.random.beta(alpha[i][j], beta[i][j]) for j in range(num_arms)] for i in range(num_players)]
        candidate_arm = [[] for _ in range(num_arms)]
        for i in range(num_players):
            lambda_prob = 0.1
            if random.random() < lambda_prob:
                choosing_result[time_slot][i] = choosing_result[time_slot - 1][i]
            else:
                if feasible_set[i]:
                    a = max(feasible_set[i], key=lambda x: theta[i][x])
                    choosing_result[time_slot][i] = a
                else:
                    choosing_result[time_slot][i] = choosing_result[time_slot - 1][i]
            candidate_arm[choosing_result[time_slot][i]].append(i)
        for i in range(num_players):
            matching = choosing_result[time_slot][i]
            player_index = [arm_to_player_preferences[matching].index(p) for p in candidate_arm[matching]]
            Min = min(player_index)
            minindex = player_index.index(Min)
            if candidate_arm[matching][minindex] == i:
                matching_result[time_slot][i] = choosing_result[time_slot][i]
                if random.random() < r[i][matching]:
                    Y = 1
                else:
                    Y = 0
                alpha[i][matching] += Y
                beta[i][matching] += 1 - Y
                regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i]) - Y
                utilities[i][time_slot] = utilities[i][time_slot - 1] + Y
            else:
                regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i])
                utilities[i][time_slot] = utilities[i][time_slot - 1]
        update_conflict(time_slot, candidate_arm, conflict_times)
        sum_conflicts += conflict_times[time_slot]
        if time_slot % 500 == 0:
            conflict_sum.append(sum_conflicts / 500)
            sum_conflicts = 0
        if k <= L:
            k += 1
        else:
            k = 1

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
    return av_regret, av_utility, conflict_times, conflict_sum

def run_market_ucb():
    choosing_result = [[-1] * num_players for _ in range(num_time_slots)]
    matching_result = [[-1] * num_players for _ in range(num_time_slots)]
    player_ucb = [[INF] * num_arms for _ in range(num_players)]
    player_es_mean = [[0] * num_arms for _ in range(num_players)]
    player_count = [[0] * num_arms for _ in range(num_players)]
    feasible_set = [[] for _ in range(num_players)]
    candidate_arm = [[] for _ in range(num_arms)]
    regrets = [[0] * num_time_slots for _ in range(num_players)]
    utilities = [[0] * num_time_slots for _ in range(num_players)]
    conflict_times = [0] * num_time_slots
    conflict_sum = []

    k = 1
    sum_conflicts = 0
    for time_slot in tqdm(range(1, num_time_slots)):
        if k == 1:
            player_ucb = [[INF] * num_arms for _ in range(num_players)]
            candidate_arm = [[] for _ in range(num_arms)]
            for i in range(num_players):
                choosing_result[time_slot][i] = random.randint(0, num_arms - 1)
            for i in range(num_players):
                matching_result[time_slot][i] = -1
                candidate_arm[choosing_result[time_slot][i]].append(i)
            for j in range(num_arms):
                player_index = []
                if not candidate_arm[j]:
                    continue
                for p_i in candidate_arm[j]:
                    player_index.append(arm_to_player_preferences[j].index(p_i))
                Min = min(player_index)
                minindex = player_index.index(Min)
                matching_result[time_slot][candidate_arm[j][minindex]] = j
            k += 1
        else:
            feasible_set = update_feasible_set(time_slot, num_players, num_arms, arm_to_player_preferences, matching_result, feasible_set)
            candidate_arm = [[] for _ in range(num_arms)]
            for i in range(num_players):
                lambda_prob = 0.1
                if random.random() < lambda_prob:
                    choosing_result[time_slot][i] = choosing_result[time_slot - 1][i]
                else:
                    if feasible_set[i]:
                        a = feasible_set[i][0]
                        for arm in feasible_set[i]:
                            if player_ucb[i][arm] > player_ucb[i][a]:
                                a = arm
                        choosing_result[time_slot][i] = a
                    else:
                        choosing_result[time_slot][i] = choosing_result[time_slot - 1][i]
                candidate_arm[choosing_result[time_slot][i]].append(i)
            for i in range(num_players):
                matching = choosing_result[time_slot][i]
                player_index = [arm_to_player_preferences[matching].index(p) for p in candidate_arm[matching]]
                Min = min(player_index)
                minindex = player_index.index(Min)
                if candidate_arm[matching][minindex] == i:
                    matching_result[time_slot][i] = choosing_result[time_slot][i]
                    if random.random() < r[i][matching]:
                        Y = 1
                    else:
                        Y = 0
                    player_es_mean[i][matching] = (player_es_mean[i][matching] * player_count[i][matching] + Y) / (player_count[i][matching] + 1)
                    player_ucb[i][matching] = player_es_mean[i][matching] + np.sqrt(3 * np.log(time_slot) / (2 * player_count[i][matching]))
                    player_count[i][matching] += 1
                    regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i]) - Y
                    utilities[i][time_slot] = utilities[i][time_slot - 1] + Y
                else:
                    regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i])
                    utilities[i][time_slot] = utilities[i][time_slot - 1]
            update_conflict(time_slot, candidate_arm, conflict_times)
            sum_conflicts += conflict_times[time_slot]
            if time_slot % 500 == 0:
                conflict_sum.append(sum_conflicts / 500)
                sum_conflicts = 0
            if k <= L:
                k += 1
            else:
                k = 1
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
    return av_regret, av_utility, conflict_times, conflict_sum

def run_market_random():
    choosing_result = [[-1] * num_players for _ in range(num_time_slots)]
    matching_result = [[-1] * num_players for _ in range(num_time_slots)]
    candidate_arm = [[] for _ in range(num_arms)]
    regrets = [[0] * num_time_slots for _ in range(num_players)]
    utilities = [[0] * num_time_slots for _ in range(num_players)]
    conflict_times = [0] * num_time_slots
    conflict_sum = []
    sum_conflicts = 0
    for time_slot in tqdm(range(1, num_time_slots)):
        candidate_arm = [[] for _ in range(num_arms)]
        for i in range(num_players):
            choosing_result[time_slot][i] = random.randint(0, num_arms - 1)
        for i in range(num_players):
            matching_result[time_slot][i] = -1
            candidate_arm[choosing_result[time_slot][i]].append(i)
        for j in range(num_arms):
            player_index = []
            if not candidate_arm[j]:
                continue
            for p_i in candidate_arm[j]:
                player_index.append(arm_to_player_preferences[j].index(p_i))
            Min = min(player_index)
            minindex = player_index.index(Min)
            matching_result[time_slot][candidate_arm[j][minindex]] = j
        for i in range(num_players):
            if not matching_result[time_slot][i]:
                regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i])
                utilities[i][time_slot] = utilities[i][time_slot - 1]
            else:
                matching = matching_result[time_slot][i]
                if random.random() < r[i][matching]:
                    Y = 1
                else:
                    Y = 0
                regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i]) - Y
                utilities[i][time_slot] = utilities[i][time_slot - 1] + Y
        update_conflict(time_slot, candidate_arm, conflict_times)
        sum_conflicts += conflict_times[time_slot]
        if time_slot % 500 == 0:
            conflict_sum.append(sum_conflicts / 500)
            sum_conflicts = 0
            
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
    return av_regret, av_utility, conflict_times, conflict_sum

def gen_r(file_path):
    r = [[0] * num_arms for _ in range(num_players)]
    df = pd.read_csv(file_path)
    data = df.iloc[1:].values.transpose()
    num_columns = data.shape[1]
    selected_columns = np.random.choice(num_columns, size=num_arms, replace=False)
    data = data[:, selected_columns]
    data[np.isnan(data)] = 0
    player_ucb = np.random.rand(num_players, data.shape[0])
    data_r = np.dot(player_ucb, data)
    max_val = np.max(data_r)
    min_val = np.min(data_r)
    for i in range(num_players):
        for j in range(num_arms):
            data_r[i][j] = (max_val - data_r[i][j]) / (max_val - min_val)
    for i in range(num_players):
        temp = sorted(data_r[i])
        t_r = [temp.index(data_r[i][j]) for j in range(num_arms)]
        r[i]= [t_r[j] / num_arms for j in range(num_arms)]
    return r

def plot_chart(chart_type, x_values, y_values, title, xlabel, ylabel, labels, colors):
    width = 0.2
    x = np.arange(len(x_values))
    if chart_type == 'bar':
        for i, (y, label, color) in enumerate(zip(y_values, labels, colors)):
            plt.bar(x + i * width, y, width=width, color=color, label=label)
    elif chart_type == 'line':
        for i, (y, label, color) in enumerate(zip(y_values, labels, colors)):
            plt.plot(x_values, y, color=color, marker='o', linestyle='-', linewidth=2, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    global r
    r = gen_r(file_path)
    
    thompson_results = run_market_thompson_sampling()
    ucb_results = run_market_ucb()
    random_results = run_market_random()
    
    time_thousand = range(100)
    time_slots = range(1, num_time_slots + 1)
    time_sum = [i for i in range(int(num_time_slots / 500) - 1)]
    labels = ['Thompson Sampling', 'UCB', 'Random']
    colors = ['blue', 'green', 'red']
    # np.linspace(0, num_time_slots, 100, dtype=int)
    plot_chart('line', time_thousand, [np.array(result[0])[np.linspace(0, num_time_slots-1, 100, dtype=int)] for result in [thompson_results, ucb_results, random_results]], 'Regret', 'Time', 'Regret', labels, colors)
    plot_chart('bar', range(20), [np.array(result[1])[np.linspace(0, num_time_slots-1, 20, dtype=int)] for result in [thompson_results, ucb_results, random_results]], 'Utility', 'Time', 'Utility', labels, colors)
    # plot_chart('line', time_thousand, [result[0][:100] for result in [thompson_results, ucb_results, random_results]], 'Regret', 'Time', 'Regret', labels, colors)
    # plot_chart('bar', time_thousand, [result[1][:100] for result in [thompson_results, ucb_results, random_results]], 'Utility', 'Time', 'Utility', labels, colors)
    plot_chart('line', time_slots, [result[2] for result in [thompson_results, ucb_results, random_results]], 'Conflict Times', 'Time', 'Conflict Times', labels, colors)
    plot_chart('line', time_sum, [result[3] for result in [thompson_results, ucb_results, random_results]], 'Conflict Sum', 'Time', 'Conflict Sum', labels, colors)

main()
