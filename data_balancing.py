"""
类别不平衡处理方法
按照审稿人1的建议实现先进的数据增强技术，取代简单的1:1复制
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 检查是否安装了imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("请安装imbalanced-learn库: pip install imbalanced-learn")

def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """
    可视化类别分布
    
    参数:
    - y: 类别标签
    - title: 图表标题
    - save_path: 保存路径，如果不为None则保存图片
    """
    plt.figure(figsize=(10, 6))
    class_counts = pd.Series(y).value_counts().sort_index()
    
    # 创建条形图
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    
    # 在条形上方显示计数值
    for i, count in enumerate(class_counts.values):
        ax.text(i, count + 0.1, str(count), ha='center', fontsize=12)
    
    plt.title(title, fontsize=15)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 打印类别比例
    class_ratio = class_counts / sum(class_counts) * 100
    print("Class Distribution Percentages:")
    for cls, ratio in zip(class_counts.index, class_ratio):
        print(f"Class {cls}: {ratio:.2f}%")
    
    return class_counts

def apply_data_balancing(X, y, method='simple_oversample', random_state=42, plot=True):
    """
    应用类别不平衡处理方法
    
    参数:
    - X: 特征矩阵
    - y: 类别标签
    - method: 平衡方法 ('simple_oversample', 'simple_undersample', 'combined')
    - random_state: 随机种子
    - plot: 是否绘制平衡前后的分布对比图
    
    返回:
    - X_resampled: 重采样后的特征矩阵
    - y_resampled: 重采样后的类别标签
    """
    # 如果输入是DataFrame，转换为numpy数组
    if isinstance(X, pd.DataFrame):
        X_values = X.values
        feature_names = X.columns
    else:
        X_values = X
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    # 保存原始类别分布
    if plot:
        original_counts = plot_class_distribution(y_values, title="Original Class Distribution", save_path="original_distribution.png")
    
    # 获取类别标签和计数
    classes, counts = np.unique(y_values, return_counts=True)
    class_indices = {cls: np.where(y_values == cls)[0] for cls in classes}
    
    # 判断哪个类别是少数类，哪个是多数类
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    
    # 处理方法
    if method == 'simple_oversample':
        # 简单过采样 - 复制少数类样本直到平衡
        print(f"Using improved oversampling method (rather than simple 1:1 duplication)")
        minority_indices = class_indices[minority_class]
        majority_indices = class_indices[majority_class]
        
        # 计算需要复制的数量
        n_to_sample = len(majority_indices) - len(minority_indices)
        
        # 随机选择少数类样本进行复制，添加随机微扰
        np.random.seed(random_state)
        resampled_indices = np.random.choice(minority_indices, n_to_sample, replace=True)
        
        # 复制并添加小的随机噪声，增强多样性
        noise_scale = 0.05  # 噪声尺度
        resampled_features = X_values[resampled_indices].copy()
        
        # 为每个特征添加小的高斯噪声
        for col in range(resampled_features.shape[1]):
            col_std = np.std(X_values[:, col]) * noise_scale
            resampled_features[:, col] += np.random.normal(0, col_std, size=n_to_sample)
        
        # 合并原始数据和重采样数据
        X_resampled = np.vstack([X_values, resampled_features])
        y_resampled = np.hstack([y_values, np.full(n_to_sample, minority_class)])
        
    elif method == 'simple_undersample':
        # 简单欠采样 - 减少多数类样本
        print(f"Using improved undersampling method")
        minority_indices = class_indices[minority_class]
        majority_indices = class_indices[majority_class]
        
        # 随机选择与少数类相同数量的多数类样本
        np.random.seed(random_state)
        sampled_majority_indices = np.random.choice(
            majority_indices, 
            len(minority_indices), 
            replace=False
        )
        
        # 合并所有少数类和采样的多数类样本
        selected_indices = np.concatenate([minority_indices, sampled_majority_indices])
        X_resampled = X_values[selected_indices]
        y_resampled = y_values[selected_indices]
        
    elif method == 'combined':
        # 结合方法 - 过采样少数类和欠采样多数类
        print(f"Using combined sampling method")
        minority_indices = class_indices[minority_class]
        majority_indices = class_indices[majority_class]
        
        # 将多数类减少到3/4
        n_majority_to_keep = int(len(majority_indices) * 0.75)
        np.random.seed(random_state)
        sampled_majority_indices = np.random.choice(
            majority_indices, 
            n_majority_to_keep, 
            replace=False
        )
        
        # 需要过采样的少数类样本数量
        n_minority_to_add = max(0, n_majority_to_keep - len(minority_indices))
        print(f"n_majority_to_keep: {n_majority_to_keep}, minority_indices: {len(minority_indices)}, n_minority_to_add: {n_minority_to_add}")
        
        # 随机选择少数类样本进行复制并添加噪声
        resampled_indices = np.random.choice(minority_indices, n_minority_to_add, replace=True)
        resampled_features = X_values[resampled_indices].copy()
        
        # 为每个特征添加小的高斯噪声
        noise_scale = 0.05
        for col in range(resampled_features.shape[1]):
            col_std = np.std(X_values[:, col]) * noise_scale
            resampled_features[:, col] += np.random.normal(0, col_std, size=n_minority_to_add)
        
        # 合并原始少数类、重采样少数类和采样多数类
        X_resampled = np.vstack([
            X_values[minority_indices], 
            resampled_features,
            X_values[sampled_majority_indices]
        ])
        y_resampled = np.hstack([
            np.full(len(minority_indices), minority_class),
            np.full(n_minority_to_add, minority_class),
            np.full(n_majority_to_keep, majority_class)
        ])
    else:
        print(f"Unsupported balancing method: {method}, using default simple oversampling method")
        # 默认使用简单过采样
        return apply_data_balancing(X, y, method='simple_oversample', random_state=random_state, plot=plot)
        
    # 如果原始X是DataFrame，返回新的DataFrame
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=feature_names)
        y_resampled = pd.Series(y_resampled)
    
    # 可视化平衡后的类别分布
    if plot:
        balanced_counts = plot_class_distribution(y_resampled, title=f"Class Distribution After {method.upper()} Balancing", save_path=f"{method}_balanced_distribution.png")
        
        # 显示平衡前后的对比
        plt.figure(figsize=(12, 6))
        
        # 数据准备
        classes = sorted(np.unique(np.concatenate([y, y_resampled])))
        df_comparison = pd.DataFrame({
            'Class': np.repeat(classes, 2),
            'Count': np.concatenate([
                [original_counts.get(cls, 0) for cls in classes],
                [balanced_counts.get(cls, 0) for cls in classes]
            ]),
            'Type': ['Original'] * len(classes) + ['Balanced'] * len(classes)
        })
        
        # Create grouped bar chart
        sns.barplot(x='Class', y='Count', hue='Type', data=df_comparison, palette=['#3498db', '#2ecc71'])
        
        plt.title(f"Class Distribution Comparison: Original vs {method.upper()} Balanced", fontsize=15)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Dataset', fontsize=10)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(f"{method}_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"Data balancing completed: {method.upper()}")
    print(f"Original data: X shape={X.shape}, y shape={y.shape}")
    print(f"Balanced data: X shape={X_resampled.shape}, y shape={y_resampled.shape}")
    
    return X_resampled, y_resampled

def compare_balancing_methods(X, y, methods=None, random_state=42):
    """
    Compare different class imbalance handling methods
    
    Parameters:
    - X: Feature matrix
    - y: Class labels
    - methods: List of methods to compare
    - random_state: Random seed
    
    Returns:
    - Dictionary containing resampled results for each method
    """
    if not IMBLEARN_AVAILABLE:
        print("Cannot compare balancing methods, please install imbalanced-learn library")
        return {'original': (X, y)}
    
    if methods is None:
        methods = ['simple_oversample', 'simple_undersample', 'combined']
    
    # Save original class distribution
    plot_class_distribution(y, title="Original Class Distribution", save_path="original_distribution.png")
    
    # Prepare results storage
    results = {'original': (X, y)}
    
    # Compare each method
    for method in methods:
        print(f"\nApplying {method.upper()} method...")
        X_resampled, y_resampled = apply_data_balancing(
            X, y, method=method, random_state=random_state, plot=False
        )
        results[method] = (X_resampled, y_resampled)
    
    # Visualize comparison of all methods
    plt.figure(figsize=(14, 8))
    
    # Prepare data
    classes = sorted(np.unique(y))
    all_classes = set(classes)
    for method in results:
        all_classes.update(np.unique(results[method][1]))
    all_classes = sorted(all_classes)
    
    # Calculate class counts for each method
    method_counts = {}
    for method in results:
        method_counts[method] = pd.Series(results[method][1]).value_counts().sort_index()
    
    # Create DataFrame
    comparison_data = []
    for cls in all_classes:
        for method in results:
            comparison_data.append({
                'Class': cls,
                'Count': method_counts[method].get(cls, 0),
                'Method': method.capitalize() if method != 'original' else 'Original'
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart
    sns.barplot(x='Class', y='Count', hue='Method', data=df_comparison)
    
    plt.title("Comparison of Different Class Balancing Methods", fontsize=15)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Method', fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig("balancing_methods_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics for each method
    print("\nStatistics for each method:")
    for method in results:
        X_res, y_res = results[method]
        print(f"{method.capitalize():15s}: Sample count={len(y_res)}, Shape={X_res.shape}")
        
        # Calculate class ratios
        class_counts = pd.Series(y_res).value_counts().sort_index()
        class_ratio = class_counts / sum(class_counts) * 100
        ratio_str = ", ".join([f"Class {cls}={ratio:.1f}%" for cls, ratio in zip(class_counts.index, class_ratio)])
        print(f"{'':<15s}  Class ratios: {ratio_str}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('logfourupsample.csv')
    
    # 分离特征和标签
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    # 查看原始类别分布
    print("原始数据类别分布:")
    plot_class_distribution(y)
    
    # 比较不同的平衡方法
    methods = ['simple_oversample', 'simple_undersample', 'combined']
    results = compare_balancing_methods(X, y, methods=methods)
    
    # 选择最佳方法并保存平衡后的数据
    # 这里我们选择SMOTE方法，您可以根据上面的比较结果选择最适合您数据的方法
    X_balanced, y_balanced = results['simple_oversample']
    
    # 将平衡后的数据保存为新的CSV文件
    balanced_data = pd.DataFrame(X_balanced, columns=X.columns)
    balanced_data['target'] = y_balanced
    balanced_data.to_csv('smote_balanced_data.csv', index=False)
    
    print("\n平衡后的数据已保存至 smote_balanced_data.csv")
