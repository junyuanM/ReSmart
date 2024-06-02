import pandas as pd
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import normalize

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

def run_market(num_players, num_arms, num_time_slots, L, arm_to_player_preferences, r, choosing_result, matching_result, alpha, beta, theta, feasible_set, candidate_arm, regrets, utilities, conflict_times, conflict_sum):
    k = 1
    h = 1
    sum_conflicts = 0
    eta = 0.1
    S = [[0] * len(L)]
    p = []
    L_h = [0]
    Z_h = [0]
    per_time = 0
    av_utility = [0] * num_time_slots

    for time_slot in tqdm(range(1, num_time_slots)):
        if k == 1:
            alpha = [[1] * num_arms for _ in range(num_players)]
            beta = [[1] * num_arms for _ in range(num_players)]
            candidate_arm = [[] for _ in range(num_arms)]
            for i in range(num_players):
                choosing_result[time_slot - 1][i] = random.randint(0, num_arms - 1)
            for i in range(num_players):
                matching_result[time_slot - 1][i] = -1
                candidate_arm[choosing_result[time_slot - 1][i]].append(i)
            for j in range(num_arms):
                if candidate_arm[j]:
                    player_index = [arm_to_player_preferences[j].index(p) for p in candidate_arm[j]]
                    Min = min(player_index)
                    minindex = player_index.index(Min)
                    matching_result[time_slot - 1][candidate_arm[j][minindex]] = j

            sum_l = sum(math.exp(eta * S[h - 1][l]) for l in range(len(L)))
            p_h = [math.exp(eta * S[h - 1][l]) / sum_l for l in range(len(L))]
            L_h.append(L_h[-1] + np.random.choice(L, size=1, p=p_h)[0])

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
        per_time += 1
        if per_time == L_h[h] - L_h[h - 1]:
            conflict_sum.append(sum_conflicts / per_time)
            sum_conflicts = 0
            per_time = 0
        if k < (L_h[h] - L_h[h - 1]):
            k += 1
        else:
            k = 1
            for j in range(L_h[h - 1] + 1, L_h[h] + 1):
                for i in range(num_players):
                    av_utility[j] += utilities[i][j]
                av_utility[j] = av_utility[j] / num_players
            av_ut_l = sum(av_utility[L_h[h - 1] + 1: L_h[h] + 1]) / (L_h[h] - L_h[h - 1])
            Z_h.append(av_ut_l)
            temp_Sh = []
            for l in range(len(L)):
                if L_h[h] - L_h[h - 1] == l:
                    temp_Shj = S[h - 1][l] + 1 - (1 - Z_h[h]) / p_h[l]
                else:
                    temp_Shj = S[h - 1][l] + 1
                temp_Sh.append(temp_Shj)
            S.append(temp_Sh)
            h += 1

def gen_r(file_path, num_players, num_arms):
    r = [[0] * num_arms for _ in range(num_players)]
    df = pd.read_csv(file_path)
    data = df.iloc[1:].values.transpose()
    num_columns = data.shape[1]
    selected_columns = np.random.choice(num_columns, size=num_arms, replace=False)
    data = data[:, selected_columns]
    data[np.isnan(data)] = 0
    theta = np.random.rand(num_players, data.shape[0])
    data_r = np.dot(theta, data)
    max_val = np.max(data_r)
    min_val = np.min(data_r)
    for i in range(num_players):
        for j in range(num_arms):
            data_r[i][j] = (data_r[i][j] - min_val) / (max_val - min_val)
    for i in range(num_players):
        temp = sorted(data_r[i])
        t_r = [temp.index(data_r[i][j]) for j in range(num_arms)]
        r[i] = [t_r[j] / num_arms for j in range(num_arms)]
    return r

def plot_chart(chart_type, x_values, y_values, title, xlabel, ylabel, labels, colors):
    for i, (y, label, color) in enumerate(zip(y_values, labels, colors)):
        if chart_type == 'bar':
            plt.bar(x_values, y, color=color, label=label)
        elif chart_type == 'line':
            plt.plot(x_values, y, color=color, marker='o', linestyle='-', linewidth=2, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def run_scenario(file_path, num_players, num_arms, num_time_slots, L):
    arm_to_player_preferences = [random.sample(range(num_players), num_players) for _ in range(num_arms)]
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
    r = gen_r(file_path, num_players, num_arms)
    run_market(num_players, num_arms, num_time_slots, L, arm_to_player_preferences, r, choosing_result, matching_result, alpha, beta, theta, feasible_set, candidate_arm, regrets, utilities, conflict_times, conflict_sum)
    av_regret = [sum(regrets[i][j] for i in range(num_players)) / num_players for j in range(num_time_slots)]
    av_utility = [sum(utilities[i][j] for i in range(num_players)) / num_players for j in range(num_time_slots)]

    return av_regret, av_utility, conflict_times, conflict_sum

def main():
    file_path = './dataset/food/new_data.csv'
    num_time_slots = 100000
    L = [i for i in range(5000, 10000, 500)]
    scenarios = [(10, 10), (25, 25), (50, 50)]
    results = [run_scenario(file_path, num_players, num_arms, num_time_slots, L) for num_players, num_arms in scenarios]
    
    time_slots = np.linspace(0, num_time_slots, 100, dtype=int)
    labels = [f'{num_players} players, {num_arms} arms' for num_players, num_arms in scenarios]
    colors = ['blue', 'green', 'red']

    plot_chart('line', range(100), [result[0][:100] for result in results], 'Regret', 'Time', 'Regret', labels, colors)
    plot_chart('bar', range(100), [result[1][:100] for result in results], 'Utility', 'Time', 'Utility', labels, colors)
    # plot_chart('line', range(100), [np.array(result[0])[np.linspace(0, num_time_slots-1, 100, dtype=int)] for result in results], 'Regret', 'Time', 'Regret', labels, colors)
    # plot_chart('bar', range(20), [np.array(result[1])[np.linspace(0, num_time_slots-1, 20, dtype=int)] for result in results], 'Utility', 'Time', 'Utility', labels, colors)
    # plot_chart('line', range(1, num_time_slots + 1), [result[2] for result in results], 'Conflict Times', 'Time', 'Conflict Times', labels, colors)
    # Ensure conflict_sum has the same length for all results
    max_len = max(len(result[3]) for result in results)
    conflict_sums_interpolated = [np.interp(np.linspace(0, len(result[3])-1, max_len), np.arange(len(result[3])), result[3]) for result in results]
    plot_chart('line', range(max_len), conflict_sums_interpolated, 'Conflict Sum', 'Time', 'Conflict Sum', labels, colors)

main()
