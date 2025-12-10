import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_samples = (10000,3)
 
mean = 5
sd = 2 # Normal distribution
N = np.random.normal(loc=mean, scale=sd, size=num_samples)

a = 0.3   #ax^(-1/a)
P = np.random.power(a, size=num_samples)

p = 0.005 # Probability of success for geometric distribution
G = np.random.geometric(p=p, size=num_samples)

# Boxplot
plt.figure(figsize=(24,6))
plt.boxplot([N[:,0], N[:,1], N[:,2], P[:,0], P[:,1], P[:,2], G[:,0], G[:,1], G[:,2]],
            tick_labels=['N1', 'N2', 'N3', 'P1', 'P2', 'P3', 'G1', 'G2', 'G3'])
plt.title('Boxplot of Random Samples from Different Distributions')
plt.ylabel('Value')
plt.show()

# Histogram
plt.figure(figsize=(24,6))
plt.hist(N.flatten(), bins=30, alpha=1, label='Normal Distribution', color='red')
plt.hist(P.flatten(), bins=30, alpha=0.5, label='Power Law Distribution', color='blue')
plt.hist(G.flatten(), bins=30, alpha=0.2, label='Geometric Distribution', color='yellow')
plt.title('Histogram of Random Samples from Different Distributions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# 1. Divide By MAX

N_max = np.max(N,axis=0)
P_max = np.max(P,axis=0)
G_max = np.max(G,axis=0)

#divide by maximum value of each variable in each distribution
N_max_divide = N / N_max
P_max_divide = P / P_max
G_max_divide = G / G_max

plt.figure(figsize=(12,6))
plt.boxplot([N_max_divide[:,0], N_max_divide[:,1], N_max_divide[:,2], P_max_divide[:,0], P_max_divide[:,1], P_max_divide[:,2], G_max_divide[:,0], G_max_divide[:,1], G_max_divide[:,2]],
            tick_labels=['N1', 'N2', 'N3', 'P1', 'P2', 'P3', 'G1', 'G2', 'G3'])
plt.title('Boxplot of Random Samples from Different Distributions After Division by Max Element')
plt.ylabel('Value')
plt.show()

# 2. Divide By SUM

N_sum = np.sum(N, axis=0)
P_sum = np.sum(P, axis=0)
G_sum = np.sum(G, axis=0)

N_sum_divide = N / N_sum
P_sum_divide = P / P_sum
G_sum_divide = G / G_sum

plt.figure(figsize=(12,6))
plt.boxplot([N_sum_divide[:,0], N_sum_divide[:,1], N_sum_divide[:,2], P_sum_divide[:,0], P_sum_divide[:,1], P_sum_divide[:,2], G_sum_divide[:,0], G_sum_divide[:,1], G_sum_divide[:,2]],
            tick_labels=['N1', 'N2', 'N3', 'P1', 'P2', 'P3', 'G1', 'G2', 'G3'])
plt.title('Boxplot of Random Samples from Different Distributions After Division by Sum of Elements')
plt.ylabel('Value')
plt.show()

# 3. Standardization into z-score

#mean and standard deviation of each distribution for each variable
n_mean = np.mean(N, axis=0)
p_mean = np.mean(P, axis=0)
g_mean = np.mean(G, axis=0)

n_std = np.std(N, axis=0)
p_std = np.std(P, axis=0)
g_std = np.std(G, axis=0)

#calculate the z-score for each variable in each distribution
n_z = (N - n_mean) / n_std
p_z = (P - p_mean) / p_std
g_z = (G - g_mean) / g_std

plt.figure(figsize=(12,6))
plt.boxplot([n_z[:,0], n_z[:,1], n_z[:,2], p_z[:,0], p_z[:,1], p_z[:,2], g_z[:,0], g_z[:,1], g_z[:,2]],
            tick_labels=['N1', 'N2', 'N3', 'P1', 'P2', 'P3', 'G1', 'G2', 'G3'])
plt.title('Boxplot of Random Samples from Different Distributions After Z-Score Normalization')
plt.ylabel('Value')
plt.show()

# 4. Percentile Normalization

# Checking how argsort Works
kdkd = np.array([1,15,6,8,11])
# argsort returns the indices of the element in the sorted array
x = np.array(kdkd).argsort()
y = np.array(kdkd).argsort().argsort()
print(x,kdkd,y)

N_x_percentiles, N_y_percentiles, N_z_percentiles = [], [], []
P_x_percentiles, P_y_percentiles, P_z_percentiles = [], [], []
G_x_percentiles, G_y_percentiles, G_z_percentiles = [], [], []


originalArray = [N[:,0], N[:,1], N[:,2],
                 P[:,0], P[:,1], P[:,2],
                 G[:,0], G[:,1], G[:,2]]

percentileArrays = [N_x_percentiles, N_y_percentiles, N_z_percentiles,
                    P_x_percentiles, P_y_percentiles, P_z_percentiles,
                    G_x_percentiles, G_y_percentiles, G_z_percentiles]

n = len(N) 

for i in range(9):
    data = originalArray[i]
    ranks = data.argsort().argsort() + 1  
    percentiles = 100 * (ranks - 1) / (n - 1)
    percentileArrays[i].extend(percentiles)

plt.figure(figsize=(12,6))
plt.boxplot([N_x_percentiles, N_y_percentiles, N_z_percentiles, P_x_percentiles, P_y_percentiles, P_z_percentiles, G_x_percentiles, G_y_percentiles, G_z_percentiles],
            tick_labels=['N1', 'N2', 'N3', 'P1', 'P2', 'P3', 'G1', 'G2', 'G3'])
plt.title('Boxplot of Random Samples from Different Distributions')
plt.ylabel('Value')
plt.show()

# 5. Median Normalization

n_median = np.median(N, axis=0)
p_median = np.median(P, axis=0)
g_median = np.median(G, axis=0)

A1 = np.array([n_median[0], p_median[0], g_median[0]])
m1 = (n_median[0] + p_median[0] + g_median[0])/3
multiplier1 = m1 / A1

A2 = np.array([n_median[1], p_median[1], g_median[1]])
m2 = (n_median[1] + p_median[1] + g_median[1])/3
multiplier2 = m2 / A2

A3 = np.array([n_median[2], p_median[2], g_median[2]])
m3 = (n_median[2] + p_median[2] + g_median[2])/3
multiplier3 = m3 / A3

mul1 = np.array([multiplier1[0], multiplier2[0], multiplier3[0]])
mul2 = np.array([multiplier1[1], multiplier2[1], multiplier3[1]])
mul3 = np.array([multiplier1[2], multiplier2[2], multiplier3[2]])

N_median_normalized = N * mul1
P_median_normalized = P * mul2
G_median_normalized = G * mul3

plt.figure(figsize=(12,6))
plt.boxplot([N_median_normalized[:,0], N_median_normalized[:,1], N_median_normalized[:,2], P_median_normalized[:,0], P_median_normalized[:,1], P_median_normalized[:,2], G_median_normalized[:,0], G_median_normalized[:,1], G_median_normalized[:,2]],
            tick_labels=['N1', 'N2', 'N3', 'P1', 'P2', 'P3', 'G1', 'G2', 'G3'])
plt.title('Boxplot of Random Samples from Different Distributions')
plt.ylabel('Value')
plt.show()

# 6. Quantile Normalization

def quantile_normalize(data):
    
    sorted_data = np.sort(data, axis=1)              
    rank_means = np.mean(sorted_data, axis=0)        

    ranks = data.argsort(axis=1).argsort(axis=1)     

    normalized_data = np.zeros_like(data, dtype=float)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            normalized_data[i,j] = rank_means[ranks[i,j]]

    return normalized_data



data = np.array( [N[:, 0],P[:, 0],G[:, 0]])  
normalized_df1 = quantile_normalize(data)

data = np.array([N[:,1],P[:,1],G[:,1]])
normalized_df2 = quantile_normalize(data)

data = np.array([N[:,2],P[:,2],G[:,2]])
normalized_df3 = quantile_normalize(data)

plt.figure(figsize=(12,6))
plt.boxplot([normalized_df1[0,:], normalized_df2[0,:], normalized_df3[0,:], 
             normalized_df1[1,:], normalized_df2[1,:], normalized_df3[1,:],
             normalized_df1[2,:], normalized_df2[2,:], normalized_df3[2,:]],
            tick_labels=['N1', 'N2', 'N3', 'P1', 'P2', 'P3', 'G1', 'G2', 'G3'])
plt.title('Boxplot of Random Samples from Different Distributions')
plt.ylabel('Value')
plt.show()



# Comparison Together

# VALUES OF NORMAL DISTRIBUTION BEFORE AND AFTER NORMALIZATION

percentile_lists = [N_x_percentiles, N_y_percentiles, N_z_percentiles]
normalized_dfs = [normalized_df1, normalized_df2, normalized_df3]

plt.figure(figsize=(18, 6))

for i in range(3):
    percentile = percentile_lists[i]
    normalized_df = normalized_dfs[i]
    
   
    plt.subplot(1,3,i+1)

    plt.hist(N[:, i], bins=30, alpha=0.1, label=f'N{i+1}', color='black')
    plt.hist(normalized_df[0,:], bins=30, alpha=0.1, label=f'N{i+1} Quantile Normalized', color='red')
    plt.hist(N_max_divide[:, i], bins=30, alpha=0.9, label=f'N{i+1} Max Divide', color='blue')
    plt.hist(N_sum_divide[:, i], bins=30, alpha=0.8, label=f'N{i+1} Sum Divide', color='green')
    plt.hist(n_z[:, i], bins=30, alpha=0.9, label=f'N{i+1} Z-Score', color='orange')
    plt.hist(N_median_normalized[:, i], bins=30, alpha=0.4, label=f'N{i+1} Median Normalized', color='purple')
    plt.hist(percentile, bins=30, alpha=0.5, label=f'N{i+1} Percentiles', color='cyan')

    plt.title(f'Histogram of Normalized N{i+1} Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()

# VALUES OF POWER LAW DISTRIBUTION BEFORE AND AFTER NORMALIZATION

percentile_lists = [P_x_percentiles, P_y_percentiles, P_z_percentiles]
normalized_dfs = [normalized_df1, normalized_df2, normalized_df3]

plt.figure(figsize=(18, 6))

for i in range(3):
    percentile = percentile_lists[i]
    normalized_df = normalized_dfs[i]
    
   
    plt.subplot(1,3,i+1)

    plt.hist(P[:, i], bins=30, alpha=0.1, label=f'P{i+1}', color='black')
    plt.hist(normalized_df[1,:], bins=30, alpha=0.1, label=f'P{i+1} Quantile Normalized', color='red')
    plt.hist(P_max_divide[:, i], bins=30, alpha=0.9, label=f'P{i+1} Max Divide', color='blue')
    plt.hist(P_sum_divide[:, i], bins=30, alpha=0.8, label=f'P{i+1} Sum Divide', color='green')
    plt.hist(p_z[:, i], bins=30, alpha=0.9, label=f'P{i+1} Z-Score', color='orange')
    plt.hist(P_median_normalized[:, i], bins=30, alpha=0.4, label=f'P{i+1} Median Normalized', color='purple')
    plt.hist(percentile, bins=30, alpha=0.5, label=f'P{i+1} Percentiles', color='cyan')

    plt.title(f'Histogram of Normalized P{i+1} Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()

# VALUES OF GEOMETRIC DISTRIBUTION BEFORE AND AFTER NORMALIZATION

percentile_lists = [G_x_percentiles, G_y_percentiles, G_z_percentiles]
normalized_dfs = [normalized_df1, normalized_df2, normalized_df3]

plt.figure(figsize=(18, 6))

for i in range(3):
    percentile = percentile_lists[i]
    normalized_df = normalized_dfs[i]
    
   
    plt.subplot(1,3,i+1)

    plt.hist(G[:, i], bins=30, alpha=0.1, label=f'G{i+1}', color='black')
    plt.hist(normalized_df[2,:], bins=30, alpha=0.1, label=f'G{i+1} Quantile Normalized', color='red')
    plt.hist(G_max_divide[:, i], bins=30, alpha=0.9, label=f'G{i+1} Max Divide', color='blue')
    plt.hist(G_sum_divide[:, i], bins=30, alpha=0.8, label=f'G{i+1} Sum Divide', color='green')
    plt.hist(g_z[:, i], bins=30, alpha=0.9, label=f'G{i+1} Z-Score', color='orange')
    plt.hist(G_median_normalized[:, i], bins=30, alpha=0.4, label=f'G{i+1} Median Normalized', color='purple')
    plt.hist(percentile, bins=30, alpha=0.5, label=f'G{i+1} Percentiles', color='cyan')

    plt.title(f'Histogram of Normalized G{i+1} Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()



# COMPARISON USING BOX PLOT


# VALUES OF NORMAL DISTRIBUTION AFTER DIFFERENT NORMALIZATION

percentile_lists = [N_x_percentiles, N_y_percentiles, N_z_percentiles]
normalized_dfs = [normalized_df1, normalized_df2, normalized_df3]
plt.figure(figsize=(18, 6))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.boxplot([N[:,i], N_max_divide[:,i], N_sum_divide[:,i], n_z[:,i], N_median_normalized[:,i], percentile_lists[i],normalized_dfs[i][0,:]],
                label=['N', 'N Max Divide', 'N Sum Divide', 'N Z-Score', 'N Median Normalized', 'N Percentiles', 'N Quantile Normalized'])
    plt.ylabel('Value')
    plt.legend()

plt.suptitle('Boxplot of Normal Distribution Before and After Normalization')
plt.tight_layout()
plt.show()    



# VALUES OF POWERLAW DISTRIBUTION AFTER DIFFERENT NORMALIZATION

percentile_lists = [P_x_percentiles, P_y_percentiles, P_z_percentiles]
normalized_dfs = [normalized_df1, normalized_df2, normalized_df3]
plt.figure(figsize=(18, 6))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.boxplot([P[:,i], P_max_divide[:,i], P_sum_divide[:,i], p_z[:,i], P_median_normalized[:,i], percentile_lists[i],normalized_dfs[i][1,:]],
                label=['P', 'P Max Divide', 'P Sum Divide', 'P Z-Score', 'P Median Normalized', 'P Percentiles', 'P Quantile Normalized'])
    plt.ylabel('Value')
    plt.legend()

plt.suptitle('Boxplot of Power Law Distribution Before and After Normalization')
plt.tight_layout()
plt.show()    



# VALUES OF GEOMETRIC DISTRIBUTION AFTER DIFFERENT NORMALIZATION

percentile_lists = [G_x_percentiles, G_y_percentiles, G_z_percentiles]
normalized_dfs = [normalized_df1, normalized_df2, normalized_df3]
plt.figure(figsize=(18, 6))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.boxplot([G[:,i], G_max_divide[:,i], G_sum_divide[:,i], g_z[:,i], G_median_normalized[:,i], percentile_lists[i],normalized_dfs[i][2,:]],
                label=['G', 'G Max Divide', 'G Sum Divide', 'G Z-Score', 'G Median Normalized', 'G Percentiles', 'G Quantile Normalized'])
    plt.ylabel('Value')
    plt.legend()

plt.suptitle('Boxplot of Geometric Distribution Before and After Normalization')
plt.tight_layout()
plt.show()    