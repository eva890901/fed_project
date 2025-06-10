import copy
import torch
from torch import nn
import numpy as np
from collections import OrderedDict

def FedAvg(current_w_locals):
    num = len(current_w_locals)
    first_client_params = next(iter(current_w_locals.values()))
    w_avg = {k: torch.zeros_like(v) for k, v in first_client_params.items()}
    for k in w_avg.keys():
        for w_locals in current_w_locals.values():
            w_avg[k] += w_locals[k]
        w_avg[k] = torch.div(w_avg[k], num)
    return w_avg

def Combine_Metrics(w_local):
    if not w_local:
        raise ValueError("Input w_local is empty. Cannot perform aggregation.")
    
    normal_weights = w_local
    print(f"Initial clients: {len(normal_weights)}")
    
    # Filter 1: K-medoids clustering
    malicious_index_filter1 = k_medoids_clustering(w_local, 2, 100, 1e-3)  # Adjusted tol
    normal_weights = {key: value for key, value in normal_weights.items() if key not in malicious_index_filter1}
    print(f"Clients after k-medoids: {len(normal_weights)}")
    
    # Filter 2: Cosine similarity
    if normal_weights:
        malicious_index_filter2 = cosine_similarity(normal_weights, 0.9)  # Adjusted threshold
        normal_weights = {key: value for key, value in normal_weights.items() if key not in malicious_index_filter2}
        print(f"Clients after cosine similarity: {len(normal_weights)}")
    else:
        print("Warning: No clients left after k-medoids clustering. Falling back to original weights.")
        return FedAvg(w_local)
    
    # Filter 3: PPMCC
    if normal_weights:
        malicious_index_filter3 = PPMCC(normal_weights, 0.7)  # Adjusted threshold
        normal_weights = {key: value for key, value in normal_weights.items() if key not in malicious_index_filter3}
        print(f"Clients after PPMCC: {len(normal_weights)}")
    else:
        print("Warning: No clients left after cosine similarity. Falling back to original weights.")
        return FedAvg(w_local)
    
    # Final aggregation
    if normal_weights:
        w_global = FedAvg(normal_weights)
    else:
        print("Warning: No clients left after all filters. Falling back to original weights.")
        w_global = FedAvg(w_local)
    
    return w_global

"""
<K-medoids Algorithm>
  1. 初始化:
      隨機挑選k(k=2)個data point作為初始的medoids
  2. 分配:
      計算每個data point到所有medoids的距離(通常會用Euclidean Distance)
      將每個data point分配到距離最近的medoid , 這樣就會分出k個cluster
  3. 更新:
      對每一個cluster , 嘗試將medoid替換成該cluster內的其他點 , 計算替換後的總成本(所有點到最近medoid的Euclidean distance)
      如果替換後的總成本降低 , 則更新該點為新的medoid
  4. 迭代:
      重複分配與更新的步驟 , 直到medoids不再變化或達到最大的迭代次數
  5. 終止:
      返回最終的medoids與對應的cluster分類
"""
def k_medoids_clustering(w_local, k, max_iter, tol):
    # 定義設備（根據是否可用 GPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 獲取客戶端索引並將權重轉為張量
    client_index = list(w_local.keys())
    weights = torch.stack([flatten_weights(w_local[i]).to(device) for i in client_index])
    num = len(client_index)
    
    # 隨機選擇初始中心點（使用 PyTorch 以保持設備一致）
    medoid_index = torch.randperm(num, device=device)[:k]
    medoids = weights[medoid_index]
    
    # 初始化標籤張量，確保在正確設備上
    labels = torch.zeros(num, dtype=torch.long, device=device)
    
    for _ in range(max_iter):
        # 計算每個點到中心點的歐幾里得距離
        euclidean_dis = torch.cdist(weights, medoids, p=2)
        new_labels = torch.argmin(euclidean_dis, dim=1)
        
        # 檢查標籤是否收斂
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        
        # 更新每個分群的中心點
        for cluster_index in range(k):
            cluster_points_index = (labels == cluster_index).nonzero(as_tuple=True)[0]
            if len(cluster_points_index) == 0:
                continue
            cluster_points = weights[cluster_points_index]
            intra_dis = torch.cdist(cluster_points, cluster_points, p=2)
            total_dis = intra_dis.sum(dim=1)
            best_medoid_index = cluster_points_index[torch.argmin(total_dis)]
            medoids[cluster_index] = weights[best_medoid_index]
    
    # 計算每個分群的大小並找出正常分群
    cluster_sizes = torch.bincount(labels, minlength=k)
    normal_cluster = torch.argmax(cluster_sizes).item()
    
    # 找出異常（惡意）客戶端索引
    malicious_client_index = [client_index[i] for i in range(num) if labels[i].item() != normal_cluster]
    
    return malicious_client_index

def cosine_similarity(weights, threshold):
    """
    計算每個客戶端權重與基準權重的平均餘弦相似度，返回相似度小於閾值的客戶端索引列表。
    
    Args:
        weights (dict): 客戶端權重字典，鍵是客戶端索引，值是權重字典（鍵是參數名稱，值是張量）。
        threshold (float): 餘弦相似度閾值。
    
    Returns:
        list: 餘弦相似度小於閾值的客戶端索引列表。
    """
    similarity_dict = {}  # 儲存每個客戶端的平均相似度
    benchmark = coordinate_wise(weights)  # 獲取基準參數（NumPy 陣列）

    for client_index, w_local in weights.items():
        total_similarity = 0.0
        total_params = 0

        for k in benchmark.keys():
            # 將客戶端權重展平為一維數組，確保張量在 CPU 上
            client_vector = np.array(w_local[k].cpu(), dtype=float).flatten()
            median_vector = np.array(benchmark[k], dtype=float).flatten()

            # 計算餘弦相似度
            dot_product = np.dot(client_vector, median_vector)
            client_norm = np.linalg.norm(client_vector)
            median_norm = np.linalg.norm(median_vector)

            if client_norm == 0 or median_norm == 0:
                total_similarity += 0.0
            else:
                total_similarity += dot_product / (client_norm * median_norm)
            total_params += 1

        # 計算平均相似度
        similarity_dict[client_index] = total_similarity / total_params if total_params > 0 else 0.0

    # 找出相似度小於閾值的客戶端索引
    malicious_index = [client_index for client_index, sim in similarity_dict.items() if sim < threshold]

    return malicious_index

def PPMCC(weights, threshold):
    correlations = {}
    benchmark = coordinate_wise(weights)
    for client_index, w_local in weights.items():
        correlation = 0.0
        total_params = 0

        for k in benchmark.keys():
            # 將客戶端權重和基準權重展平為一維數組，確保張量在 CPU 上
            client_vector = np.array(w_local[k].cpu(), dtype=float).flatten()  # 添加 .cpu()
            median_vector = np.array(benchmark[k], dtype=float).flatten()  # benchmark 已為 NumPy 陣列，無需 .cpu()
        
            if len(client_vector) < 2:
                # 向量長度小於 2，無法計算相關係數
                correlation += 0.0
            else:
                # 減去平均值
                client_vector_adj = client_vector - np.mean(client_vector)
                median_vector_adj = median_vector - np.mean(median_vector)
                    
                # 計算協方差和標準差
                covariance = np.sum(client_vector_adj * median_vector_adj) / (len(client_vector) - 1)
                std_client = np.std(client_vector, ddof=1)  # ddof=1 使用 n-1
                std_median = np.std(median_vector, ddof=1)
                    
                # 避免除以零
                if std_client == 0 or std_median == 0:
                    correlation += 0.0
                else:
                    correlation += covariance / (std_client * std_median)
            total_params += 1
        correlations[client_index] = correlation / total_params if total_params > 0 else 0.0
    malicious_index = [client_index for client_index, c in correlations.items() if c < threshold]

    return malicious_index

def coordinate_wise(weights):
    # 將字典的值轉換為 OrderedDict 列表
    dict_list = list(weights.values())

    # 檢查是否為空
    if not dict_list:
        raise ValueError("Client parameters dictionary is empty")

    # 獲取參數鍵（假設所有客戶端的 OrderedDict 有相同的鍵）
    keys = dict_list[0].keys()

    # 檢查所有客戶端是否有相同的鍵
    for d in dict_list:
        if d.keys() != keys:
            raise ValueError(f"Inconsistent parameter keys: {d.keys()} vs {keys}")

    result = OrderedDict()

    for key in keys:
        # 提取所有客戶端的該鍵數值，轉為 NumPy 數組，確保張量在 CPU 上
        values = [
            np.array(d[key].cpu(), dtype=float) if isinstance(d[key], torch.Tensor) else d[key]
            for d in dict_list
        ]

        # 檢查形狀一致性
        if not all(v.shape == values[0].shape for v in values):
            raise ValueError(f"Inconsistent shapes for key {key}: {[v.shape for v in values]}")

        shape = values[0].shape
        median_values = np.zeros(shape)

        if len(shape) == 1:  # 一維
            for i in range(shape[0]):
                position_values = [v[i] for v in values]
                median = np.median(position_values)
                median_values[i] = int(median) if median.is_integer() else median
        else:  # 多維
            for idx in np.ndindex(shape):
                position_values = [v[idx] for v in values]
                median = np.median(position_values)
                median_values[idx] = int(median) if median.is_integer() else median

        result[key] = median_values  # 直接儲存 NumPy 陣列，cosine_similarity 可直接使用

    return result

def flatten_weights(w_local):
    # 將 OrderedDict 的權重展平為單一張量
    flattened = []
    for key in w_local:
        flattened.append(w_local[key].flatten())
    return torch.cat(flattened)