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
        mi = matching_result[time_slot - 1][i]
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

def run_market_etc():
    choosing_result = [[-1] * num_players for _ in range(num_time_slots)]
    matching_result = [[-1] * num_players for _ in range(num_time_slots)]
    player_es_mean = [[0] * num_arms for _ in range(num_players)]
    player_count = [[0] * num_arms for _ in range(num_players)]
    candidate_arm = [[] for _ in range(num_arms)]
    regrets = [[0] * num_time_slots for _ in range(num_players)]
    utilities = [[0] * num_time_slots for _ in range(num_players)]
    conflict_times = [0] * num_time_slots
    conflict_sum = []

    h = 1  # Exploration parameter
    sum_conflicts = 0
    for time_slot in tqdm(range(1, num_time_slots)):
        if time_slot <= h * num_arms:
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
                matching = matching_result[time_slot][i]
                if matching != -1:
                    if random.random() < r[i][matching]:
                        Y = 1
                    else:
                        Y = 0
                    player_es_mean[i][matching] = (player_es_mean[i][matching] * player_count[i][matching] + Y) / (player_count[i][matching] + 1)
                    player_count[i][matching] += 1
                    regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i]) - Y
                    utilities[i][time_slot] = utilities[i][time_slot - 1] + Y
                else:
                    regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i])
                    utilities[i][time_slot] = utilities[i][time_slot - 1]
        elif time_slot == h * num_arms + 1:
            matchij = [0] * num_arms
            for i in range(num_players):
                sorted_idx = np.argsort(player_es_mean[i])[::-1]
                for p in range(num_arms):
                    if matchij[sorted_idx[p]] == 0:
                        matchij[sorted_idx[p]] = 1
                        choosing_result[time_slot][i] = sorted_idx[p]
                        matching_result[time_slot][i] = sorted_idx[p]
                        matching = matching_result[time_slot][i]
                        if random.random() < r[i][matching]:
                            Y = 1
                        else:
                            Y = 0
                        regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i]) - Y
                        utilities[i][time_slot] = utilities[i][time_slot - 1] + Y
                        player_es_mean[i][p] = (player_es_mean[i][p] * player_count[i][p] + Y) / (player_count[i][p] + 1)
                        player_count[i][p] += 1
                        break
            candidate_arm = [[] for _ in range(num_arms)]
            for i in range(num_players):
                candidate_arm[choosing_result[time_slot][i]].append(i)
        else:
            for i in range(num_players):
                choosing_result[time_slot][i] = choosing_result[time_slot - 1][i]
                matching_result[time_slot][i] = matching_result[time_slot - 1][i]
                matching = matching_result[time_slot][i]
                if random.random() < r[i][matching]:
                    Y = 1
                else:
                    Y = 0
                regrets[i][time_slot] = regrets[i][time_slot - 1] + min(r[i]) - Y
                utilities[i][time_slot] = utilities[i][time_slot - 1] + Y
                player_es_mean[i][matching] = (player_es_mean[i][matching] * player_count[i][matching] + Y) / (player_count[i][matching] + 1)
                player_count[i][matching] += 1
            candidate_arm = [[] for _ in range(num_arms)]
            for i in range(num_players):
                candidate_arm[choosing_result[time_slot][i]].append(i)
        update_conflict(time_slot, candidate_arm, conflict_times)
        sum_conflicts += conflict_times[time_slot]
        if time_slot % 500 == 0:
            conflict_sum.append(sum_conflicts / 500)
            sum_conflicts = 0

    av_regret = [0] * num_time_slots
    for j in range(num_time_slots):
        for i in range(num_players):
            av_regret[j] += regrets[i][j]
        av_regret[j] = av_regret[j] / num_players

    av_utility = [0] * num_time_slots
    for j in range(num_time_slots):
        for i in range(num_players):
            av_utility[j] += utilities[i][j]
        av_utility[j] = av_utility[j] / num_players

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
        r[i] = [t_r[j] / num_arms for j in range(num_arms)]
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
    etc_results = run_market_etc()
    
    time_thousand = range(100)
    time_slots = range(1, num_time_slots + 1)
    time_sum = [i for i in range(int(num_time_slots / 500) - 1)]
    labels = ['Thompson Sampling', 'UCB', 'Random', 'ETC']
    colors = ['blue', 'green', 'red', 'purple']

    plot_chart('line', time_thousand, [np.array(result[0])[np.linspace(0, num_time_slots-1, 100, dtype=int)] for result in [thompson_results, ucb_results, random_results, etc_results]], 'Regret', 'Time', 'Regret', labels, colors)
    plot_chart('bar', range(20), [np.array(result[1])[np.linspace(0, num_time_slots-1, 20, dtype=int)] for result in [thompson_results, ucb_results, random_results, etc_results]], 'Utility', 'Time', 'Utility', labels, colors)
    plot_chart('line', time_slots, [result[2] for result in [thompson_results, ucb_results, random_results, etc_results]], 'Conflict Times', 'Time', 'Conflict Times', labels, colors)
    plot_chart('line', time_sum, [result[3] for result in [thompson_results, ucb_results, random_results, etc_results]], 'Conflict Sum', 'Time', 'Conflict Sum', labels, colors)

main()
