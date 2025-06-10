import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib
# Comment out Agg backend for interactive environments if needed
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
def Draw_Distribution(num_classes, classes, labels, client_idxs, num_users, alpha):
    plt.figure(figsize=(10, 7))
    plt.hist([labels[idx] for idx in client_idxs], stacked=True, bins=np.arange(min(labels)-0.5, max(labels)+1.5, 1),
             label=["Client {}".format(i+1) for i in range(num_users)], rwidth=0.5)
    plt.xticks(np.arange(num_classes), classes, rotation=45, ha='center')
    plt.xlabel("Label Type")
    plt.ylabel("Number of Samples")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 0.2), borderaxespad=-2.25, frameon=False, fontsize=10)
    plt.subplots_adjust(right=0.75, bottom=0.2)
    plt.title(f"MNIST Label Distribution (alpha={alpha}) on Different Clients")
    plt.savefig(f"./result/MNIST_alpha_{alpha}.png")
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

def Draw_Distribution(num_classes, classes, labels, client_idxs, num_users, alpha):
    # 如果 labels 是 PyTorch 張量，則轉換為 NumPy 數組
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()  # 轉換為 NumPy 數組
    
    # 確保 client_idxs 是每個客戶端的索引列表
    client_labels = []
    for idxs in client_idxs:
        # 提取當前客戶端的標籤
        client_labels.append([labels[idx] for idx in idxs])
    
    # 繪製直方圖
    plt.figure()
    plt.hist(client_labels, stacked=True, bins=np.arange(min(labels)-0.5, max(labels)+1.5, 1))
    plt.title(f'Dirichlet Distribution (alpha={alpha})')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.savefig(f'./dirichlet_alpha_{alpha}.png')
    plt.close()
    
def Dirichlet_Distribution(classes, train_dataset, alpha, num_users):
    num_classes = len(classes)  # Number of classes

    train_labels = train_labels = np.array([target for _, target in train_dataset])

    if alpha == float('inf'):
        # Uniform distribution across clients for each class
        label_distribution = np.full((num_classes, num_users), 1.0 / num_users)
    else:
        # Dirichlet distribution for non-uniform splitting
        label_distribution = np.random.dirichlet([alpha] * num_users, num_classes)

    # Get indices for each class
    class_idxs = [np.where(train_labels == y)[0] for y in range(num_classes)]

    # Initialize client indices
    client_idxs = [[] for _ in range(num_users)]

    # Distribute samples for each class across clients
    for c, fracs in zip(class_idxs, label_distribution):
        # Ensure fractions sum to 1 and handle splits
        if len(c) == 0:  # Skip if no samples for this class
            continue
        split_indices = (np.cumsum(fracs)[:-1] * len(c)).astype(int)
        split_indices = np.clip(split_indices, 0, len(c))  # Ensure valid splits
        for i, idxs in enumerate(np.split(c, split_indices)):
            client_idxs[i].extend(idxs.tolist())

    # Convert lists to numpy arrays
    client_idxs = [np.array(idxs) for idxs in client_idxs]

    # Draw distribution with alpha
    Draw_Distribution(num_classes, classes, train_labels, client_idxs, num_users, alpha)

    return client_idxs
"""
# Main
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)

# Get class labels (0 to 9 for MNIST)
dataset_classes = np.unique(dataset_train.targets).tolist()  # MNIST has 10 classes: 0 to 9

# Extract labels as numpy array
train_labels = np.array([target for _, target in dataset_train])

# Generate client distribution
train_client_distribution = Dirichlet_Distribution(dataset_classes, train_labels, alpha=0.5, num_users=5)
"""