import numpy as np
from scipy.stats import ttest_1samp
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.datasets import *

# 数据加载
data = load_iris()
X = data.data
y = data.target

# 数据预处理：归一化
def normalize(X0):
    X = X0.copy()
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
    return X

data2 = normalize(X)

# 阿尔法贝塔散度函数
def alpha_beta_divergence(p, q, alpha, beta):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    eps = 1e-10  # 避免零值
    p = p + eps
    q = q + eps
    term1 = np.sum(p**alpha * q**beta)
    term2 = (alpha / (alpha + beta)) * np.sum(p**(alpha + beta))
    term3 = (beta / (alpha + beta)) * np.sum(q**(alpha + beta))
    div = - (1 / (alpha * beta)) * (term1 - term2 - term3)
    return div

# 对称阿尔法贝塔散度（用于DBSCAN）
def symmetric_alpha_beta_divergence(p, q, alpha=0.5, beta=1.1):
    d_pq = alpha_beta_divergence(p, q, alpha, beta)
    d_qp = alpha_beta_divergence(q, p, alpha, beta)
    return (d_pq + d_qp) / 2
    
def ab_divergence_metric(p, q):
    return alpha_beta_divergence(p, q, alpha=0.5, beta=1.4)

# DBSCAN聚类
alpha = 0.5
beta = 1.4
eps = 0.1  # 需要根据数据调整
min_samples = 5

ab_distance = lambda p, q: symmetric_alpha_beta_divergence(p, q, alpha=alpha, beta=beta)
db = DBSCAN(eps=eps, min_samples=min_samples, metric=ab_divergence_metric).fit(data2)
clusters = db.labels_
centers = db.core_sample_indices_

# 处理噪声点（-1），映射为新类别
if -1 in clusters:
    max_label = max(clusters[clusters != -1])
    clusters[clusters == -1] = max_label + 1

# 性能评估
f1 = f1_score(y, clusters, average='macro')
acc = accuracy_score(y, clusters)
nmi = mutual_info_score(y, clusters)
ari = adjusted_rand_score(y, clusters)
ami = adjusted_mutual_info_score(y, clusters)

print(f"F-measure: {f1}")
print(f"Accuracy: {acc}")
print(f"NMI: {nmi}")
print(f"ARI: {ari}")
print(f"AMI: {ami}")

# 可视化
def visualize_clusters(data, clusters, centers):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    unique_clusters = set(clusters)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    for i, cluster in enumerate(unique_clusters):
        plt.scatter(data_2d[clusters == cluster, 0], data_2d[clusters == cluster, 1], color=colors[i], label=f'Cluster {cluster}', alpha=0.7)
    for center in centers:
        plt.scatter(data_2d[center, 0], data_2d[center, 1], color='red', marker='x', s=100, label='Center' if center == centers[0] else "")
    plt.legend()
    plt.show()

visualize_clusters(data2, clusters, centers)


n_clusters = 2
n_runs = 30
accuracies_kmeans = []
for _ in range(n_runs):
    kmeans = KMeans(n_clusters=n_clusters)
    labels_kmeans = kmeans.fit_predict(data2)  # 使用归一化数据
    acc = accuracy_score(y, labels_kmeans)
    accuracies_kmeans.append(acc)
t_stat, p_value = ttest_1samp(accuracies_kmeans, popmean=acc)
print(f"p-value (vs k-means): {p_value:.4f}")
s_numpy = np.std(accuracies_kmeans, ddof=1)  # ddof=1表示计算样本标准差
print(f"样本标准差 (Numpy): {s_numpy:.2f}%")
s_numpy = np.std(acc, ddof=1)  # ddof=1表示计算样本标准差
print(f"样本标准差 (Numpy): {s_numpy:.2f}%")