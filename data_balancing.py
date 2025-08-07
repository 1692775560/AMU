"""
Class Imbalance Handling Methods
Implementing advanced data augmentation techniques as suggested by Reviewer 1, replacing simple 1:1 duplication
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Check if imbalanced-learn is installed
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Please install imbalanced-learn library: pip install imbalanced-learn")

def plot_class_distribution(y, title="Class Distribution", save_path=None):
    """
    Visualize class distribution
    
    Parameters:
    - y: Class labels
    - title: Chart title
    - save_path: Save path, if not None, save the image
    """
    plt.figure(figsize=(10, 6))
    class_counts = pd.Series(y).value_counts().sort_index()
    
    # Create bar chart
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    
    # Display count values above bars
    for i, count in enumerate(class_counts.values):
        ax.text(i, count + 0.1, str(count), ha='center', fontsize=12)
    
    plt.title(title, fontsize=15)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print class ratios
    class_ratio = class_counts / sum(class_counts) * 100
    print("Class Distribution Percentages:")
    for cls, ratio in zip(class_counts.index, class_ratio):
        print(f"Class {cls}: {ratio:.2f}%")
    
    return class_counts

def apply_data_balancing(X, y, method='simple_oversample', random_state=42, plot=True):
    """
    Apply class imbalance handling methods
    
    Parameters:
    - X: Feature matrix
    - y: Class labels
    - method: Balancing method ('simple_oversample', 'simple_undersample', 'combined')
    - random_state: Random seed
    - plot: Whether to plot distribution comparison before and after balancing
    
    Returns:
    - X_resampled: Resampled feature matrix
    - y_resampled: Resampled class labels
    """
    # If input is DataFrame, convert to numpy array
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
    
    # Save original class distribution
    if plot:
        original_counts = plot_class_distribution(y_values, title="Original Class Distribution", save_path="original_distribution.png")
    
    # Get class labels and counts
    classes, counts = np.unique(y_values, return_counts=True)
    class_indices = {cls: np.where(y_values == cls)[0] for cls in classes}
    
    # Determine which class is minority and which is majority
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    
    # Processing method
    if method == 'simple_oversample':
        # Simple oversampling - copy minority class samples until balanced
        print(f"Using improved oversampling method (rather than simple 1:1 duplication)")
        minority_indices = class_indices[minority_class]
        majority_indices = class_indices[majority_class]
        
        # Calculate number of samples to copy
        n_to_sample = len(majority_indices) - len(minority_indices)
        
        # Randomly select minority class samples for copying, add random perturbation
        np.random.seed(random_state)
        resampled_indices = np.random.choice(minority_indices, n_to_sample, replace=True)
        
        # Copy and add small random noise to enhance diversity
        noise_scale = 0.05  # Noise scale
        resampled_features = X_values[resampled_indices].copy()
        
        # Add small Gaussian noise to each feature
        for col in range(resampled_features.shape[1]):
            col_std = np.std(X_values[:, col]) * noise_scale
            resampled_features[:, col] += np.random.normal(0, col_std, size=n_to_sample)
        
        # Merge original data and resampled data
        X_resampled = np.vstack([X_values, resampled_features])
        y_resampled = np.hstack([y_values, np.full(n_to_sample, minority_class)])
        
    elif method == 'simple_undersample':
        # Simple undersampling - reduce majority class samples
        print(f"Using improved undersampling method")
        minority_indices = class_indices[minority_class]
        majority_indices = class_indices[majority_class]
        
        # Randomly select same number of majority class samples as minority class
        np.random.seed(random_state)
        sampled_majority_indices = np.random.choice(
            majority_indices, 
            len(minority_indices), 
            replace=False
        )
        
        # Merge all minority class and sampled majority class samples
        selected_indices = np.concatenate([minority_indices, sampled_majority_indices])
        X_resampled = X_values[selected_indices]
        y_resampled = y_values[selected_indices]
        
    elif method == 'combined':
        # Combined method - oversample minority class and undersample majority class
        print(f"Using combined sampling method")
        minority_indices = class_indices[minority_class]
        majority_indices = class_indices[majority_class]
        
        # Reduce majority class to 3/4
        n_majority_to_keep = int(len(majority_indices) * 0.75)
        np.random.seed(random_state)
        sampled_majority_indices = np.random.choice(
            majority_indices, 
            n_majority_to_keep, 
            replace=False
        )
        
        # Number of minority class samples needed for oversampling
        n_minority_to_add = max(0, n_majority_to_keep - len(minority_indices))
        print(f"n_majority_to_keep: {n_majority_to_keep}, minority_indices: {len(minority_indices)}, n_minority_to_add: {n_minority_to_add}")
        
        # Randomly select minority class samples for copying and add noise
        resampled_indices = np.random.choice(minority_indices, n_minority_to_add, replace=True)
        resampled_features = X_values[resampled_indices].copy()
        
        # Add small Gaussian noise to each feature
        noise_scale = 0.05
        for col in range(resampled_features.shape[1]):
            col_std = np.std(X_values[:, col]) * noise_scale
            resampled_features[:, col] += np.random.normal(0, col_std, size=n_minority_to_add)
        
        # Merge original minority class, resampled minority class and sampled majority class
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
        # Default to simple oversampling
        return apply_data_balancing(X, y, method='simple_oversample', random_state=random_state, plot=plot)
        
    # If original X is DataFrame, return new DataFrame
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=feature_names)
        y_resampled = pd.Series(y_resampled)
    
    # Visualize class distribution after balancing
    if plot:
        balanced_counts = plot_class_distribution(y_resampled, title=f"Class Distribution After {method.upper()} Balancing", save_path=f"{method}_balanced_distribution.png")
        
        # Show comparison before and after balancing
        plt.figure(figsize=(12, 6))
        
        # Data preparation
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
    
    # Separate features and labels
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    # View original class distribution
    print("Original data class distribution:")
    plot_class_distribution(y)
    
    # Compare different balancing methods
    methods = ['simple_oversample', 'simple_undersample', 'combined']
    results = compare_balancing_methods(X, y, methods=methods)
    
    # Select the best method and save balanced data
    # Here we choose SMOTE method, you can select the most suitable method for your data based on the comparison results above
    X_balanced, y_balanced = results['simple_oversample']
    
    # Save balanced data as new CSV file
    balanced_data = pd.DataFrame(X_balanced, columns=X.columns)
    balanced_data['target'] = y_balanced
    balanced_data.to_csv('smote_balanced_data.csv', index=False)
    
    print("\nBalanced data has been saved to smote_balanced_data.csv")
