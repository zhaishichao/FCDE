import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter


class BorderSMOTE:
    """
    BorderSMOTE算法实现
    
    BorderSMOTE是SMOTE的改进版本，它只在少数类样本的边界附近生成合成样本，
    而不是在整个特征空间中生成。这样可以生成更真实的合成样本。
    
    参数：
    --------
    k_neighbors : int, 默认=5
        用于找到k个最近邻的参数
    
    borderline_type : str, 默认='borderline1'
        BorderSMOTE的变体：
        - 'borderline1': 只在边界附近合成样本
        - 'borderline2': 使用不同的边界定义方式
    
    random_state : int, 默认=None
        随机数种子，用于复现结果
    
    属性：
    --------
    synthetic_samples_ : ndarray
        生成的合成样本
    """
    
    def __init__(self, k_neighbors=5, borderline_type='borderline1', random_state=None):
        self.k_neighbors = k_neighbors
        self.borderline_type = borderline_type
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def _identify_borderline_samples(self, X_minority, X_majority):
        """
        识别少数类中的边界样本
        
        参数：
        --------
        X_minority : ndarray, shape (n_minority, n_features)
            少数类样本
        
        X_majority : ndarray, shape (n_majority, n_features)
            多数类样本
        
        返回：
        --------
        borderline_indices : ndarray
            边界样本的索引
        """
        n_features = X_minority.shape[1]
        
        # 构建少数类内部的k-NN
        nbrs_minority = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X_minority))).fit(X_minority)
        distances, indices = nbrs_minority.kneighbors(X_minority)
        
        # 构建多数类的k-NN
        nbrs_majority = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(X_majority))).fit(X_majority)
        distances_majority, _ = nbrs_majority.kneighbors(X_minority)
        
        borderline_indices = []
        
        if self.borderline_type == 'borderline1':
            # Borderline-1: 少数类样本至少有一个k-NN来自多数类
            for i in range(len(X_minority)):
                # 计算k-NN中有多少个多数类样本
                # 这里通过检查最近的多数类样本距离与最近少数类样本距离的关系
                if len(distances[i]) > 1:
                    minority_nn_dist = distances[i, 1]  # 最近的少数类样本距离（跳过自己）
                else:
                    minority_nn_dist = distances[i, 0] if len(distances[i]) > 0 else float('inf')
                    
                majority_nn_dist = distances_majority[i, 0]  # 最近的多数类样本距离
                
                # 如果少数类样本靠近多数类样本，则为边界样本
                if majority_nn_dist <= minority_nn_dist:
                    borderline_indices.append(i)
        
        elif self.borderline_type == 'borderline2':
            # Borderline-2: 少数类样本的k-NN中，多数类样本数量 > 0
            for i in range(len(X_minority)):
                # 统计k-NN中多数类样本的数量
                if len(distances[i]) > 1:
                    minority_nn_dist = distances[i, 1]
                else:
                    minority_nn_dist = distances[i, 0] if len(distances[i]) > 0 else float('inf')
                    
                majority_nn_dists = distances_majority[i]  # k个最近的多数类样本
                
                # 计算有多少个多数类的k-NN距离小于少数类的最近邻距离
                count = np.sum(majority_nn_dists < minority_nn_dist)
                
                if count > 0:
                    borderline_indices.append(i)
        
        return np.array(borderline_indices, dtype=int)
    
    def _generate_synthetic_samples(self, X_minority, borderline_indices, n_synthetic):
        """
        为边界样本生成合成少数类样本
        
        参数：
        --------
        X_minority : ndarray, shape (n_minority, n_features)
            少数类样本
        
        borderline_indices : ndarray
            边界样本的索引
        
        n_synthetic : int
            要生成的合成样本数量
        
        返回：
        --------
        synthetic_samples : ndarray, shape (n_synthetic, n_features)
            生成的合成样本
        """
        synthetic_samples = []
        
        if len(borderline_indices) == 0:
            return np.array(synthetic_samples)
        
        # 只使用边界样本来找k-NN
        X_borderline = X_minority[borderline_indices]
        
        # 调整k值，确保不超过样本数
        k_use = min(self.k_neighbors, len(X_borderline) - 1)
        
        if k_use <= 0:
            # 如果边界样本太少，降级处理
            k_use = 1
        
        nbrs = NearestNeighbors(n_neighbors=k_use + 1).fit(X_borderline)
        distances, indices = nbrs.kneighbors(X_borderline)
        
        # 随机选择边界样本和它们的邻居来生成合成样本
        for _ in range(n_synthetic):
            # 随机选择一个边界样本
            random_borderline_idx = np.random.randint(0, len(X_borderline))
            random_borderline_sample = X_borderline[random_borderline_idx]
            
            # 从该样本的k-NN中随机选择一个（跳过自己）
            knn_indices = indices[random_borderline_idx, 1:]  # 跳过自己
            if len(knn_indices) > 0:
                random_knn_idx = np.random.choice(knn_indices)
                random_knn_sample = X_borderline[random_knn_idx]
            else:
                # 如果没有邻居，随机选择任何边界样本
                random_knn_idx = np.random.randint(0, len(X_borderline))
                random_knn_sample = X_borderline[random_knn_idx]
            
            # 生成合成样本
            lambda_val = np.random.random()
            synthetic_sample = random_borderline_sample + lambda_val * (
                random_knn_sample - random_borderline_sample
            )
            synthetic_samples.append(synthetic_sample)
        
        return np.array(synthetic_samples)
    
    def fit_resample(self, X, y):
        """
        对数据进行BorderSMOTE过采样，返回完整的平衡数据集
        
        参数：
        --------
        X : ndarray, shape (n_samples, n_features)
            特征矩阵
        
        y : ndarray, shape (n_samples,)
            标签向量
        
        返回：
        --------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            过采样后的特征矩阵
        
        y_resampled : ndarray, shape (n_samples_new,)
            过采样后的标签向量
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # 统计各类别数量
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        n_majority = class_counts[majority_class]
        n_minority = class_counts[minority_class]
        
        # 分离多数类和少数类
        X_majority = X[y == majority_class]
        X_minority = X[y == minority_class]
        
        # 计算需要生成的合成样本数量
        n_synthetic = n_majority - n_minority
        
        # 识别边界样本
        borderline_indices = self._identify_borderline_samples(X_minority, X_majority)
        
        print(f"识别到 {len(borderline_indices)} 个边界样本（总共 {len(X_minority)} 个少数类样本）")
        
        # 生成合成样本
        if len(borderline_indices) > 1:
            synthetic_samples = self._generate_synthetic_samples(
                X_minority, borderline_indices, n_synthetic
            )
        else:
            # 如果边界样本太少，使用所有少数类样本生成
            if len(borderline_indices) == 0:
                print("警告：未识别到边界样本，使用所有少数类样本生成合成样本")
            else:
                print("警告：边界样本过少，使用所有少数类样本生成合成样本")
            synthetic_samples = self._generate_synthetic_samples(
                X_minority, np.arange(len(X_minority)), n_synthetic
            )
        
        print(f"生成了 {len(synthetic_samples)} 个合成样本")
        
        # 合并原始少数类和合成样本
        X_minority_augmented = np.vstack([X_minority, synthetic_samples])
        y_minority_augmented = np.hstack([
            np.full(len(X_minority), minority_class),
            np.full(len(synthetic_samples), minority_class)
        ])
        
        # 合并多数类和增强后的少数类
        X_resampled = np.vstack([X_majority, X_minority_augmented])
        y_resampled = np.hstack([
            np.full(len(X_majority), majority_class),
            y_minority_augmented
        ])
        
        # 保存合成样本供后续使用
        self.synthetic_samples_ = synthetic_samples
        
        return X_resampled, y_resampled
    
    def fit_resample_only_synthetic(self, X, y):
        """
        仅返回生成的合成少数类样本
        
        参数：
        --------
        X : ndarray, shape (n_samples, n_features)
            特征矩阵
        
        y : ndarray, shape (n_samples,)
            标签向量
        
        返回：
        --------
        X_synthetic : ndarray, shape (n_synthetic, n_features)
            生成的合成样本特征矩阵
        
        y_synthetic : ndarray, shape (n_synthetic,)
            生成的合成样本标签向量
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # 统计各类别数量
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        n_majority = class_counts[majority_class]
        n_minority = class_counts[minority_class]
        
        # 分离多数类和少数类
        X_majority = X[y == majority_class]
        X_minority = X[y == minority_class]
        
        # 计算需要生成的合成样本数量
        n_synthetic = n_majority - n_minority
        
        # 识别边界样本
        borderline_indices = self._identify_borderline_samples(X_minority, X_majority)
        
        print(f"识别到 {len(borderline_indices)} 个边界样本（总共 {len(X_minority)} 个少数类样本）")
        
        # 生成合成样本
        if len(borderline_indices) > 1:
            synthetic_samples = self._generate_synthetic_samples(
                X_minority, borderline_indices, n_synthetic
            )
        else:
            # 如果边界样本太少，使用所有少数类样本生成
            if len(borderline_indices) == 0:
                print("警告：未识别到边界样本，使用所有少数类样本生成合成样本")
            else:
                print("警告：边界样本过少，使用所有少数类样本生成合成样本")
            synthetic_samples = self._generate_synthetic_samples(
                X_minority, np.arange(len(X_minority)), n_synthetic
            )
        
        print(f"生成了 {len(synthetic_samples)} 个合成样本")
        
        # 保存合成样本供后续使用
        self.synthetic_samples_ = synthetic_samples
        
        # 返回仅合成样本
        y_synthetic = np.full(len(synthetic_samples), minority_class)
        
        return synthetic_samples, y_synthetic


# 示例使用
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # 创建不平衡数据集
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        weights=[0.9, 0.1],  # 9:1不平衡
        random_state=42
    )
    
    print("原始数据集类别分布:")
    print(f"类别0: {np.sum(y == 0)}, 类别1: {np.sum(y == 1)}")
    print(f"类别比例: {np.sum(y == 0) / len(y):.2%} vs {np.sum(y == 1) / len(y):.2%}\n")
    
    # 应用BorderSMOTE - fit_resample方法
    print("=" * 50)
    print("使用 fit_resample 方法")
    print("=" * 50)
    border_smote = BorderSMOTE(k_neighbors=5, borderline_type='borderline1', random_state=42)
    X_resampled, y_resampled = border_smote.fit_resample(X, y)
    
    print("\n过采样后的数据集类别分布:")
    print(f"类别0: {np.sum(y_resampled == 0)}, 类别1: {np.sum(y_resampled == 1)}")
    print(f"类别比例: {np.sum(y_resampled == 0) / len(y_resampled):.2%} vs {np.sum(y_resampled == 1) / len(y_resampled):.2%}")
    
    # 应用BorderSMOTE - fit_resample_only_synthetic方法
    print("\n" + "=" * 50)
    print("使用 fit_resample_only_synthetic 方法")
    print("=" * 50)
    border_smote2 = BorderSMOTE(k_neighbors=5, borderline_type='borderline1', random_state=42)
    X_synthetic, y_synthetic = border_smote2.fit_resample_only_synthetic(X, y)
    
    print("\n仅合成样本的数量:")
    print(f"合成样本数: {len(X_synthetic)}")
    print(f"标签: {np.unique(y_synthetic)}")
    
    # 可视化（2D数据）
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始数据
    axes[0].scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0', alpha=0.6)
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', alpha=0.6)
    axes[0].set_title('Original Dataset')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 过采样后的数据
    axes[1].scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], 
                    label='Class 0', alpha=0.6)
    axes[1].scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], 
                    label='Class 1', alpha=0.6)
    axes[1].set_title('After BorderSMOTE')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 仅合成样本
    axes[2].scatter(X[y == 0, 0], X[y == 0, 1], label='Original Majority', alpha=0.4, s=30)
    axes[2].scatter(X[y == 1, 0], X[y == 1, 1], label='Original Minority', alpha=0.6, s=30)
    axes[2].scatter(X_synthetic[:, 0], X_synthetic[:, 1], label='Synthetic Samples', 
                    alpha=0.7, s=50, marker='^', color='red')
    axes[2].set_title('Synthetic Samples Only')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/border_smote_visualization.png', dpi=100, bbox_inches='tight')
    print("\nVisualization saved to: border_smote_visualization.png")
    plt.show()
