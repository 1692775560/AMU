"""
改进的热力图可视化
解决审稿人对Figure 2中热力图的疑问
- 清晰解释基因选择方法
- 明确标注的含义
- 增加聚类分析以支持标注
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ttest_ind

def select_informative_genes(data, label, n_genes=20, method='statistical'):
    """
    选择最具信息量的基因
    
    参数:
    - data: 基因表达矩阵 (样本 × 基因)
    - label: 样本标签
    - n_genes: 要选择的基因数量
    - method: 选择方法 ('statistical', 'variance', 'pca')
    
    返回:
    - 选择的基因列表及其重要性分数
    """
    if method == 'statistical':
        # 使用t检验找出两组间差异显著的基因
        p_values = []
        fold_changes = []
        gene_names = data.columns
        
        for gene in gene_names:
            group1 = data[gene][label == 0]
            group2 = data[gene][label == 1]
            
            # 检查数据是否足够
            if len(group1) < 2 or len(group2) < 2:
                p_values.append(1.0)
                fold_changes.append(0.0)
                continue
                
            # t检验
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            p_values.append(p_value)
            
            # 计算fold change
            mean1 = group1.mean()
            mean2 = group2.mean()
            # 避免除以零
            if mean1 == 0:
                mean1 = 1e-10
            if mean2 == 0:
                mean2 = 1e-10
            fc = np.log2(mean2 / mean1)
            fold_changes.append(fc)
        
        # 合并结果
        gene_stats = pd.DataFrame({
            'Gene': gene_names,
            'P_Value': p_values,
            'Log2FC': fold_changes
        })
        
        # 计算调整后的p值
        gene_stats['Adj_P_Value'] = gene_stats['P_Value'] * len(gene_names)  # 简单的Bonferroni校正
        gene_stats['Adj_P_Value'] = gene_stats['Adj_P_Value'].clip(upper=1.0)
        
        # 计算综合得分 (-log10(p) * |log2FC|)
        gene_stats['Score'] = -np.log10(gene_stats['P_Value']) * np.abs(gene_stats['Log2FC'])
        
        # 根据得分排序并选择前n个
        top_genes = gene_stats.sort_values('Score', ascending=False).head(n_genes)
        
        return top_genes['Gene'].tolist(), top_genes
    
    elif method == 'variance':
        # 基于方差选择
        gene_var = data.var()
        top_genes = gene_var.sort_values(ascending=False).head(n_genes)
        return top_genes.index.tolist(), pd.DataFrame({'Gene': top_genes.index, 'Score': top_genes.values})
    
    elif method == 'pca':
        # 基于PCA载荷选择
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        pca = PCA(n_components=2)
        pca.fit(scaled_data)
        
        # 获取第一主成分的载荷
        loadings = pd.DataFrame({
            'Gene': data.columns,
            'PC1_loading': pca.components_[0],
            'PC2_loading': pca.components_[1],
        })
        
        # 计算载荷的绝对值和
        loadings['Score'] = np.abs(loadings['PC1_loading']) + np.abs(loadings['PC2_loading'])
        
        # 选择载荷最高的基因
        top_genes = loadings.sort_values('Score', ascending=False).head(n_genes)
        
        return top_genes['Gene'].tolist(), top_genes
    
    else:
        raise ValueError(f"不支持的方法: {method}")

def identify_gene_clusters(data, n_clusters=3, method='kmeans'):
    """
    对基因进行聚类分析
    
    参数:
    - data: 基因表达矩阵 (转置后，基因 × 样本)
    - n_clusters: 聚类数量
    - method: 聚类方法 ('kmeans', 'hierarchical')
    
    返回:
    - 基因聚类标签
    """
    if method == 'kmeans':
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        gene_clusters = kmeans.fit_predict(data)
        return gene_clusters
    
    elif method == 'hierarchical':
        # 层次聚类
        hc = AgglomerativeClustering(n_clusters=n_clusters)
        gene_clusters = hc.fit_predict(data)
        return gene_clusters
    
    else:
        raise ValueError(f"不支持的方法: {method}")

def identify_sample_clusters(data, n_clusters=2, method='kmeans'):
    """
    对样本进行聚类分析
    
    参数:
    - data: 基因表达矩阵 (样本 × 基因)
    - n_clusters: 聚类数量
    - method: 聚类方法 ('kmeans', 'hierarchical')
    
    返回:
    - 样本聚类标签
    """
    if method == 'kmeans':
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        sample_clusters = kmeans.fit_predict(data)
        return sample_clusters
    
    elif method == 'hierarchical':
        # 层次聚类
        hc = AgglomerativeClustering(n_clusters=n_clusters)
        sample_clusters = hc.fit_predict(data)
        return sample_clusters
    
    else:
        raise ValueError(f"不支持的方法: {method}")

def plot_improved_heatmap(data1, data2=None, labels=None, title1="Gene Expression Heatmap 1", 
                          title2="Gene Expression Heatmap 2", gene_selection_method='statistical', 
                          n_genes=20, cluster_genes=True, cluster_samples=True, 
                          annotation=True, save_path=None, figsize=(30, 15)):
    """
    绘制改进的热力图，包括基因选择、聚类和明确的标注
    
    参数:
    - data1: 第一个数据集
    - data2: 第二个数据集 (可选)
    - labels: 样本标签
    - title1, title2: 热力图标题
    - gene_selection_method: 基因选择方法
    - n_genes: 要显示的基因数量
    - cluster_genes: 是否对基因进行聚类
    - cluster_samples: 是否对样本进行聚类
    - annotation: 是否添加注释
    - save_path: 保存路径
    - figsize: 图像大小
    """
    # 确保数据是DataFrame
    if not isinstance(data1, pd.DataFrame):
        data1 = pd.DataFrame(data1)
    
    if data2 is not None and not isinstance(data2, pd.DataFrame):
        data2 = pd.DataFrame(data2)
    
    # 如果没有提供标签，创建全零标签
    if labels is None:
        labels = np.zeros(len(data1))
    
    # 选择最具信息量的基因
    selected_genes, gene_stats = select_informative_genes(
        data1, labels, n_genes=n_genes, method=gene_selection_method
    )
    
    # 提取选择的基因
    data1_selected = data1[selected_genes]
    
    # 如果有第二个数据集，也提取相同的基因
    if data2 is not None:
        data2_selected = data2[selected_genes]
    
    # 对基因和样本进行聚类
    if cluster_genes:
        # 转置数据以便对基因聚类
        gene_clusters = identify_gene_clusters(data1_selected.T, n_clusters=3, method='hierarchical')
        
        # 重新排序基因
        gene_order = np.argsort(gene_clusters)
        selected_genes = [selected_genes[i] for i in gene_order]
        data1_selected = data1_selected[selected_genes]
        
        if data2 is not None:
            data2_selected = data2_selected[selected_genes]
    
    if cluster_samples:
        # 对样本聚类
        sample_clusters = identify_sample_clusters(data1_selected, n_clusters=2, method='hierarchical')
        
        # 重新排序样本
        sample_order = np.argsort(sample_clusters)
        data1_selected = data1_selected.iloc[sample_order]
        
        if data2 is not None and len(data2_selected) == len(data1_selected):
            data2_selected = data2_selected.iloc[sample_order]
    
    # 创建画布和子图
    if data2 is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
        axes = [ax]
    
    # 绘制第一张热力图
    sns.heatmap(
        data1_selected,
        cmap="mako",
        annot=False,
        fmt=".2f",
        linewidths=0.5,
        linecolor='lightgray',
        cbar_kws={"shrink": 0.8, "label": "Expression Level"},
        square=True,
        ax=axes[0]
    )
    
    # 添加标题和标签
    axes[0].set_title(title1, fontsize=20, pad=20, fontweight='bold')
    axes[0].set_xlabel('Selected Genes', fontsize=15, labelpad=15, fontweight='bold')
    axes[0].set_ylabel('Samples', fontsize=15, labelpad=15, fontweight='bold')
    
    # 设置刻度标签
    axes[0].set_xticklabels(selected_genes, rotation=90, fontsize=10, horizontalalignment='center')
    
    # 隐藏部分标签（每隔n个显示一个）
    skip = max(1, len(selected_genes) // 10)
    for i, label in enumerate(axes[0].get_xticklabels()):
        if i % skip != 0:
            label.set_visible(False)
    
    # 调整颜色条
    cbar1 = axes[0].collections[0].colorbar
    cbar1.ax.tick_params(labelsize=12)
    cbar1.set_label('Expression Level', fontsize=14, rotation=270, labelpad=20, fontweight='bold')
    
    # 基于聚类添加注释
    if annotation and cluster_genes:
        # 找出基因聚类的边界
        cluster_boundaries = []
        prev_cluster = gene_clusters[gene_order[0]]
        
        for i, idx in enumerate(gene_order):
            curr_cluster = gene_clusters[idx]
            if curr_cluster != prev_cluster:
                cluster_boundaries.append(i)
                prev_cluster = curr_cluster
        
        # 添加垂直线标示不同基因簇
        for boundary in cluster_boundaries:
            axes[0].axvline(x=boundary, color='white', linestyle='-', linewidth=2)
        
        # 标记重要区域（假设第一个簇是重要区域）
        # 这里我们可以基于基因得分确定最重要的区域
        important_genes = gene_stats.sort_values('Score', ascending=False).head(5)['Gene'].tolist()
        important_indices = [selected_genes.index(gene) for gene in important_genes if gene in selected_genes]
        
        if important_indices:
            # 计算重要区域的中心
            center_x = sum(important_indices) / len(important_indices)
            center_y = len(data1_selected) / 2
            
            # 添加箭头和标签
            axes[0].annotate(
                'Important Region\n(Differential Expression)',
                xy=(center_x, center_y),
                xytext=(center_x + len(selected_genes)/5, center_y + len(data1_selected)/5),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                fontsize=12,
                color='red',
                fontweight='bold',
                ha='center'
            )
    
    # 如果有第二个数据集，绘制第二张热力图
    if data2 is not None:
        sns.heatmap(
            data2_selected,
            cmap="rocket",
            annot=False,
            fmt=".2f",
            linewidths=0.5,
            linecolor='lightgray',
            cbar_kws={"shrink": 0.8, "label": "Expression Level"},
            square=True,
            ax=axes[1]
        )
        
        # 添加标题和标签
        axes[1].set_title(title2, fontsize=20, pad=20, fontweight='bold')
        axes[1].set_xlabel('Selected Genes', fontsize=15, labelpad=15, fontweight='bold')
        axes[1].set_ylabel('Samples', fontsize=15, labelpad=15, fontweight='bold')
        
        # 设置刻度标签
        axes[1].set_xticklabels(selected_genes, rotation=90, fontsize=10, horizontalalignment='center')
        
        # 隐藏部分标签
        for i, label in enumerate(axes[1].get_xticklabels()):
            if i % skip != 0:
                label.set_visible(False)
        
        # 调整颜色条
        cbar2 = axes[1].collections[0].colorbar
        cbar2.ax.tick_params(labelsize=12)
        cbar2.set_label('Expression Level', fontsize=14, rotation=270, labelpad=20, fontweight='bold')
        
        # 基于样本聚类添加注释
        if annotation and cluster_samples:
            # 找出样本聚类区域
            if 'sample_clusters' in locals():
                unique_clusters = np.unique(sample_clusters)
                
                # 找出第二个簇的中心区域
                key_cluster_indices = np.where(sample_clusters == 1)[0]
                if len(key_cluster_indices) > 0:
                    # 计算关键区域的中心
                    center_y = sum(key_cluster_indices) / len(key_cluster_indices)
                    center_x = len(selected_genes) / 2
                    
                    # 添加箭头和标签
                    axes[1].annotate(
                        'Key Cluster\n(Treatment Response)',
                        xy=(center_x, center_y),
                        xytext=(center_x - len(selected_genes)/5, center_y - len(data2_selected)/5),
                        arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=8),
                        fontsize=12,
                        color='blue',
                        fontweight='bold',
                        ha='center'
                    )
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图形
    plt.show()
    
    # 返回选择的基因和它们的统计信息
    return selected_genes, gene_stats

def plot_gene_importance(gene_stats, n_genes=20, title="Top Genes by Importance Score", save_path=None):
    """
    绘制基因重要性柱状图
    
    参数:
    - gene_stats: 包含基因重要性的DataFrame
    - n_genes: 要显示的基因数量
    - title: 图表标题
    - save_path: 保存路径
    """
    # 选择前n个基因
    top_genes = gene_stats.sort_values('Score', ascending=False).head(n_genes)
    
    # 创建柱状图
    plt.figure(figsize=(12, 8))
    
    # 绘制柱状图
    bars = plt.bar(range(len(top_genes)), top_genes['Score'], color='teal')
    
    # 设置x轴标签
    plt.xticks(range(len(top_genes)), top_genes['Gene'], rotation=90)
    
    # 添加标题和轴标签
    plt.title(title, fontsize=15, pad=20)
    plt.xlabel('Gene', fontsize=12, labelpad=10)
    plt.ylabel('Importance Score', fontsize=12, labelpad=10)
    
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 美化图表
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 加载数据
    try:
        data1 = pd.read_csv('logfourupsample.csv',sep=',')
        data2 = pd.read_csv('four.csv')
        
        # 提取特征和标签
        X1, y1 = data1.iloc[:, :-1], data1.iloc[:, -1]
        
        # 尝试不同的基因选择方法并绘制热力图
        selected_genes, gene_stats = plot_improved_heatmap(
            X1, data2,
            labels=y1,
            title1="Pre-treatment Gene Expression (Treatment Group)",
            title2="Post-treatment Gene Expression",
            gene_selection_method='statistical',
            n_genes=20,
            cluster_genes=True,
            cluster_samples=True,
            annotation=True,
            save_path="improved_heatmaps.png",
            figsize=(30, 15)
        )
        
        # 绘制基因重要性
        plot_gene_importance(
            gene_stats,
            n_genes=20,
            title="Top 20 Differentially Expressed Genes",
            save_path="gene_importance.png"
        )
        
        # 将选择的基因及其重要性保存到CSV文件
        gene_stats.to_csv('selected_genes_stats.csv', index=False)
        print(f"Selected {len(selected_genes)} genes and saved importance statistics to selected_genes_stats.csv")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        
        # 创建示例数据以便测试
        print("Using randomly generated data for testing...")
        np.random.seed(42)
        n_samples, n_genes = 100, 160
        
        # 创建随机数据
        random_data1 = pd.DataFrame(
            np.random.randn(n_samples, n_genes),
            columns=[f'Gene_{i+1}' for i in range(n_genes)]
        )
        random_data2 = pd.DataFrame(
            np.random.randn(n_samples, n_genes),
            columns=[f'Gene_{i+1}' for i in range(n_genes)]
        )
        
        # 创建随机标签
        random_labels = np.random.randint(0, 2, n_samples)
        
        # 使用随机数据绘制热力图
        plot_improved_heatmap(
            random_data1, random_data2,
            labels=random_labels,
            title1="Example Heatmap 1 (Random Data)",
            title2="Example Heatmap 2 (Random Data)",
            gene_selection_method='variance',
            n_genes=20,
            cluster_genes=True,
            cluster_samples=True,
            annotation=True,
            save_path="example_heatmaps.png"
        )
