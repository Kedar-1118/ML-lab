import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

def min_var_change_count(k, N):
    var_change_counts = []

    for _ in range(k):
        array = np.random.rand(N)
        # array = np.random.randint(0, 100, size=N)  # Random integers in [0, 100)
        temp_min = float('inf')
        var_change_count = 0
        i = 0
        while i < N:
            if array[i] < temp_min:
                temp_min = array[i]
                var_change_count += 1
            i += 1
        var_change_counts.append(var_change_count)


    plt.hist(var_change_counts, bins=20, edgecolor='black',color = "red")
    plt.title(f'Histogram of minimum change count (N={N})')
    plt.xlabel('var_change_count')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    
    mean_vcc = np.mean(var_change_counts)
    median_vcc = np.median(var_change_counts)
    geo_mean_vcc = gmean(var_change_counts)
    std_dev_vcc = np.std(var_change_counts)

    print(f"N = {N}")
    print(f"Mean: {mean_vcc}")
    print(f"Median: {median_vcc}")
    print(f"Geometric Mean: {geo_mean_vcc}")

    return mean_vcc, std_dev_vcc


N_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
            10000, 20000, 50000, 100000, 200000, 500000, 1000000]

mean_vcc_list = []
std_dev_list = []

for N in N_values:
    mean_vcc,std_dev = min_var_change_count(100, N)
    mean_vcc_list.append(mean_vcc)
    std_dev_list.append(std_dev)

for i in range(len(mean_vcc_list)):
    print(f"N = {N_values[i]}, Mean var_change_count = {mean_vcc_list[i]}, Std Dev = {std_dev_list[i]}")


plt.figure(figsize=(20, 12))
plt.plot(N_values, mean_vcc_list, marker='o', color='red')
plt.xscale('log', base=10)
plt.xlabel('Array Size N (log scale)')
plt.ylabel('Mean var_change_count')
plt.title('N vs Mean var_change_count')
plt.grid(True)
plt.show()

plt.figure(figsize=(20, 12))
plt.errorbar(N_values, mean_vcc_list, yerr=std_dev_list, fmt='-o', color='red', ecolor='orange', capsize=2)
plt.xscale('log', base=10)
plt.xlabel('Array Size N (log scale)')
plt.ylabel('Mean var_change_count')
plt.title('N vs Mean var_change_count with Error Bars')
plt.grid(True)
plt.show()
