import numpy as np
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
def normalize(X0):
    X = np.zeros_like(X0)
    for i in range(X0.shape[1]):
        mi, ma = np.min(X0[:, i]), np.max(X0[:, i])
        X[:, i] = (X0[:, i] - mi) / (ma - mi)
    return X
def normalize_to_distribution(data):
    """
    将数据点归一化，使得每个数据点的特征之和为1
    """
    return data / data.sum(axis=1, keepdims=True)

def standardize(data):
    """
    将数据进行标准化：减去均值，除以标准差
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std

def alpha_beta_divergence(p, q, alpha, beta):
    """
    计算两个向量 (p) 和 (q) 的 alpha-beta 发散
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # 避免负值和零值的计算问题
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    
    if alpha == beta:
        return np.sum((p**alpha - q**alpha) / alpha)
    
    a = (p**alpha - p**beta) / (alpha - beta)
    b = (q**alpha - q**beta) / (alpha - beta)
    
    return np.sum(a - b)


# def compute_density(data, alpha, beta):
#     n_samples = data.shape[0]
    
#     densities = np.zeros(n_samples)
    
#     for i in range(n_samples):
#         distances = [alpha_beta_divergence(data[i], data[j], alpha, beta) for j in range(n_samples)]
#         densities[i] = np.mean(distances)
    
#     return densities

def compute_density(data, alpha, beta, k):
    n_samples = data.shape[0]
    densities = np.zeros(n_samples)
    
    for i in range(n_samples):
        # 计算与所有其他点的散度
        divergences = []
        for j in range(n_samples):
            if i != j:
                div = alpha_beta_divergence(data[i], data[j], alpha, beta)
                divergences.append(div)
        
        # 排序并选择k个最小散度
        if len(divergences) > k:
            smallest_k = sorted(divergences)[:k]
        else:
            smallest_k = divergences
        
        # 计算k个最小散度的均值作为密度
        if smallest_k:
            densities[i] = np.mean(smallest_k)
        else:
            densities[i] = 0  # 通常不会发生
    
    return densities

def dpc_with_alpha_beta(data, alpha, beta, distance_threshold):
    data = np.asarray(data)
    densities = compute_density(data, alpha, beta,2)
    sorted_indices = np.argsort(-densities)

    clusters = [-1] * len(data)  # 初始化所有点为未分类状态
    cluster_id = 0
    cluster_centers = []

    for i in sorted_indices:
        if clusters[i] == -1:  # 如果点i未被分配到任何簇
            clusters[i] = cluster_id
            cluster_centers.append(i)

            # 遍历所有数据点，检查是否可以加入当前簇
            for j in range(len(data)):
                if clusters[j] == -1:  # 点j尚未分类
                    # 检查与当前簇内所有点的相似度
                    cluster_points = [data[k] for k in range(len(data)) if clusters[k] == cluster_id]
                    if all(alpha_beta_divergence(data[j], cp, alpha, beta) < distance_threshold for cp in cluster_points):
                        clusters[j] = cluster_id

            cluster_id += 1  # 更新簇ID
    
    return clusters, cluster_centers

def clustering_indicators(labels_true, labels_pred):
    # 如果标签为非整数（例如文本类型），将其转换为数字标签
    if type(labels_true[0]) != int:
        labels_true = LabelEncoder().fit_transform(labels_true)
    
    # F-measure (这里使用 F1 score，'macro' 表示宏平均)
    f_measure = f1_score(labels_true, labels_pred, average='macro')
    
    # Accuracy (ACC)
    accuracy = accuracy_score(labels_true, labels_pred)
    
    # Normalized Mutual Information (NMI)
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)
    
    # Rand Index (RI)
    rand_index = rand_score(labels_true, labels_pred)
    
    # Adjusted Rand Index (ARI)
    ARI = adjusted_rand_score(labels_true, labels_pred)
    
    # Adjusted Mutual Information (AMI)
    AMI = adjusted_mutual_info_score(labels_true, labels_pred)
    
    return f_measure, accuracy, normalized_mutual_information, rand_index, ARI, AMI

# 示例数据

pic=pd.read_csv("/Users/ruanenjie/学习/数据集/iris.csv", header=0)
df = pic  # 设置要读取的数据集
columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
features = columns[:len(columns) - 1]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
data = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
attributes = len(df.columns) - 1  # 属性数量（数据集维度）
original_labels = list(df[columns[-1]])  # 原始标签
data=data.to_numpy()
# data2 = normalize_to_distribution(data)  # 归一化数据，使得每个数据点的特征之和为1
data2=normalize(data)
# 调用函数
# alpha = 0.5
# beta =1.4
# distance_threshold=0.2 iris达到dpc效果的最佳参数
alpha = 0.5
beta =1.4
distance_threshold = 0.2
clusters, centers = dpc_with_alpha_beta(data2, alpha, beta, distance_threshold)
# 评估结果  
labels_pred = clusters
# 调用函数并打印结果
f_measure, accuracy, normalized_mutual_information, rand_index, ARI, AMI = clustering_indicators(original_labels, labels_pred)
print("F-measure:", f_measure)
print("Accuracy:", accuracy)
print("Rand Index:", rand_index)
print("Normalized Mutual Information:", normalized_mutual_information)
print("Adjusted Rand Index (ARI):", ARI)
print("Adjusted Mutual Information (AMI):", AMI)

# 打印簇中心的数量
print("Cluster Centers:", len(centers))

def visualize_clusters(data, clusters, centers):
    """
    可视化聚类结果
    """
    # 将数据降维到2D以便可视化
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
    else:
        reduced_data = data

    plt.figure(figsize=(10, 7))
    
    # 绘制每个簇的点
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_points = reduced_data[np.array(clusters) == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
    
    # 绘制簇中心
    center_points = reduced_data[centers]
    plt.scatter(center_points[:, 0], center_points[:, 1], c='red', marker='x', s=200, label='Centers')
    
    plt.title("Cluster Visualization")
    plt.legend()
    plt.xlabel('PC1' if data.shape[1] > 2 else 'Feature 1')
    plt.ylabel('PC2' if data.shape[1] > 2 else 'Feature 2')
    plt.grid(True)
    plt.show()

visualize_clusters(data, clusters, centers)

