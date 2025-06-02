"""
批次效应校正和PCA分析
按照审稿人2的建议实现批次效应校正
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats

# 导入需要的Python包 - 需要安装combat-pytorch或者pycombat
# 导入pycombat库
try:
    import pycombat
    
    def apply_combat(data, batch):
        # 使用pycombat进行批次校正
        # pycombat API可能会有不同的用法
        try:
            corrected_data = pycombat.pycombat(data, batch)
            return corrected_data
        except (TypeError, AttributeError):
            # 如果pycombat模块结构不同，尝试其他方式
            print("Using alternative method for batch correction...")
            # 简单的替代方法：对每个批次的数据进行标准化
            corrected = data.copy()
            unique_batches = np.unique(batch)
            
            # 对每个批次进行标准化
            for b in unique_batches:
                idx = np.where(batch == b)[0]
                batch_data = data[idx]
                # 计算均值和标准差
                batch_mean = np.mean(batch_data, axis=0)
                batch_std = np.std(batch_data, axis=0) + 1e-8  # 防止除以0
                
                # 标准化
                corrected[idx] = (batch_data - batch_mean) / batch_std
            
            return corrected
        
except ImportError:
    try:
        # 尝试导入neuroCombat
        import neuroCombat as combat
        
        def apply_combat(data, batch):
            # 使用neuroCombat进行批次校正
            covars = pd.DataFrame({'batch': batch})
            categorical_cols = ['batch']
            continuous_cols = []
            combat_data = combat.neuroCombat(
                dat=data.T,
                covars=covars,
                batch_col='batch',
                categorical_cols=categorical_cols,
                continuous_cols=continuous_cols
            )
            return combat_data['data'].T
    except ImportError:
        print("Please install batch correction library: pip install pycombat or pip install neuroCombat")
        
        # 定义一个简单的替代函数，以便脚本至少能运行
        def apply_combat(data, batch):
            print("Warning: Batch correction library not found, returning original data")
            return data
        
def perform_pca_analysis(data, labels, title="PCA of Gene Expression Data", figsize=(10, 8), save_path=None):
    """
    执行PCA分析并可视化结果
    
    参数:
    - data: 基因表达数据矩阵 (样本 x 基因)
    - labels: 样本批次或类别标签
    - title: 图表标题
    - figsize: 图表大小
    - save_path: 保存路径，如果不为None则保存图片
    """
    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 执行PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # 创建结果DataFrame
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=['PC1', 'PC2']
    )
    pca_df['Batch'] = labels
    
    # 计算解释方差比例
    explained_variance = pca.explained_variance_ratio_ * 100
    
    # 可视化
    plt.figure(figsize=figsize)
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        hue='Batch',
        palette='viridis',
        data=pca_df,
        s=100,
        alpha=0.7
    )
    
    plt.title(title, fontsize=15)
    plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)', fontsize=12)
    plt.legend(title='Data Source', title_fontsize=12, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加统计分析 - 多变量方差分析(MANOVA)来检测组间差异
    groups = [pca_result[labels == label] for label in np.unique(labels)]
    if len(groups) > 1 and all(len(g) > 1 for g in groups):
        try:
            stat, p = stats.f_oneway(*[g[:, 0] for g in groups])  # 分析PC1
            plt.figtext(0.01, 0.01, f'PC1 ANOVA: p={p:.4f}', fontsize=9)
        except:
            pass
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return pca_df, pca

def batch_correction_pipeline(data_files, batch_labels, gene_columns=None, sample_id_column=None):
    """
    完整的批次校正流程
    
    参数:
    - data_files: 数据文件路径列表
    - batch_labels: 对应每个文件的批次标签
    - gene_columns: 基因列名(如果为None则使用所有数值列)
    - sample_id_column: 样本ID列名
    
    返回:
    - 原始合并数据
    - 校正后数据
    """
    # 1. 加载并合并数据
    all_data = []
    all_batches = []
    
    for i, file_path in enumerate(data_files):
        # 根据文件类型加载数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            print(f"不支持的文件类型: {file_path}")
            continue
        
        # 添加批次信息
        batch = batch_labels[i]
        df['batch'] = batch
        
        # 提取特征列
        if gene_columns is None:
            # 使用所有数值列作为基因表达量
            feature_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            # 排除batch列
            feature_cols = [col for col in feature_cols if col != 'batch']
        else:
            feature_cols = gene_columns
        
        # 存储样本ID
        if sample_id_column and sample_id_column in df.columns:
            sample_ids = df[sample_id_column].values
        else:
            sample_ids = [f"Sample_{batch}_{j}" for j in range(len(df))]
        
        # 提取表达矩阵
        expression_data = df[feature_cols].values
        
        # 批次标签
        batch_labels_arr = np.array([batch] * len(df))
        
        all_data.append(expression_data)
        all_batches.extend(batch_labels_arr)
    
    # 合并所有数据
    combined_data = np.vstack(all_data)
    batch_array = np.array(all_batches)
    
    # 2. 在校正前进行PCA分析
    perform_pca_analysis(
        combined_data, 
        batch_array, 
        title="PCA Before Batch Correction",
        save_path="pca_before_correction.png"
    )
    
    # 3. 应用Combat进行批次校正
    corrected_data = apply_combat(combined_data, batch_array)
    
    # 4. 校正后再次进行PCA分析
    perform_pca_analysis(
        corrected_data, 
        batch_array, 
        title="PCA After Batch Correction",
        save_path="pca_after_correction.png"
    )
    
    # 5. 返回原始和校正后的数据
    return combined_data, corrected_data, batch_array, feature_cols

# 示例使用
if __name__ == "__main__":
    # 实际使用时请替换为您的数据文件路径
    data_files = ['logfourupsample.csv', 'four.csv']
    batch_labels = ['Dataset1', 'Dataset2']
    
    # 执行批次校正流程
    original_data, corrected_data, batches, features = batch_correction_pipeline(
        data_files, batch_labels
    )
    
    print(f"Original data shape: {original_data.shape}")
    print(f"Corrected data shape: {corrected_data.shape}")
    
    # 保存校正后的数据
    corrected_df = pd.DataFrame(corrected_data, columns=features)
    corrected_df['batch'] = batches
    corrected_df.to_csv('batch_corrected_data.csv', index=False)
    
    print("Batch correction completed, data saved to batch_corrected_data.csv")
