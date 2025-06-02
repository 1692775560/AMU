import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# Set the style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['savefig.dpi'] = 300

print('Figure generation script started...')

# Create output directory if it doesn't exist
output_dir = 'figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 批次效应校正前后PCA图 (Batch effect correction visualization)
try:
    raw = pd.read_csv('four.csv', index_col=0)
    print('Successfully loaded four.csv')
    
    # Try to load the corrected data if available
    try:
        # First try the expected name
        try:
            corrected = pd.read_csv('four_corrected.csv', index_col=0)
            has_corrected = True
            print('Successfully loaded four_corrected.csv')
        except FileNotFoundError:
            # Try the alternative filename
            corrected = pd.read_csv('batch_corrected_data.csv', index_col=0)
            has_corrected = True
            print('Successfully loaded batch_corrected_data.csv')
    except FileNotFoundError:
        has_corrected = False
        print('Warning: Neither four_corrected.csv nor batch_corrected_data.csv found. Only original data PCA will be plotted.')
    
    # Check for batch column
    if 'batch' in raw.columns:
        batch_info_raw = raw['batch']
    else:
        batch_info_raw = None
        print('Warning: No batch column found in original data')
    
    # Calculate PCA for original data
    features_raw = raw.drop(['response', 'batch'], axis=1, errors='ignore')
    pca_raw_model = PCA(n_components=2)
    pca_raw = pca_raw_model.fit_transform(features_raw.values)
    var_explained_raw = pca_raw_model.explained_variance_ratio_ * 100
    
    if has_corrected:
        # Prepare corrected data for PCA
        if 'batch' in corrected.columns and 'response' in corrected.columns:
            features_corr = corrected.drop(['response', 'batch'], axis=1, errors='ignore')
            batch_info_corr = corrected['batch']
        else:
            features_corr = corrected
            batch_info_corr = batch_info_raw  # Use the same batch info if not in corrected data
        
        # Calculate PCA for corrected data
        pca_corr_model = PCA(n_components=2)
        pca_corr = pca_corr_model.fit_transform(features_corr.values)
        var_explained_corr = pca_corr_model.explained_variance_ratio_ * 100
        
        # Plot both original and corrected data PCAs with batch coloring
        plt.figure(figsize=(15, 6))
        
        # Original data PCA
        plt.subplot(1, 2, 1)
        if batch_info_raw is not None:
            batches = batch_info_raw.unique()
            cmap = plt.cm.get_cmap('viridis', len(batches))
            
            for i, batch in enumerate(batches):
                batch_indices = batch_info_raw == batch
                plt.scatter(pca_raw[batch_indices, 0], pca_raw[batch_indices, 1], 
                           c=[cmap(i)], label=f'Batch {batch}', alpha=0.7, edgecolors='none')
            plt.legend(title='Batch')
        else:
            plt.scatter(pca_raw[:, 0], pca_raw[:, 1], c='blue', alpha=0.7, edgecolors='none')
            
        plt.title('PCA Before Batch Correction', fontweight='bold')
        plt.xlabel(f'Principal Component 1 ({var_explained_raw[0]:.1f}%)')
        plt.ylabel(f'Principal Component 2 ({var_explained_raw[1]:.1f}%)')
        plt.grid(True, alpha=0.3)
        
        # Annotate batch effect
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        plt.annotate('Visible batch effect', xy=(x_min + (x_max-x_min)*0.05, y_min + (y_max-y_min)*0.05), 
                     xycoords='data', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
        # Corrected data PCA
        plt.subplot(1, 2, 2)
        if batch_info_corr is not None:
            batches = batch_info_corr.unique()
            cmap = plt.cm.get_cmap('viridis', len(batches))
            
            for i, batch in enumerate(batches):
                batch_indices = batch_info_corr == batch
                plt.scatter(pca_corr[batch_indices, 0], pca_corr[batch_indices, 1], 
                           c=[cmap(i)], label=f'Batch {batch}', alpha=0.7, edgecolors='none')
            plt.legend(title='Batch')
        else:
            plt.scatter(pca_corr[:, 0], pca_corr[:, 1], c='green', alpha=0.7, edgecolors='none')
            
        plt.title('PCA After Batch Correction', fontweight='bold')
        plt.xlabel(f'Principal Component 1 ({var_explained_corr[0]:.1f}%)')
        plt.ylabel(f'Principal Component 2 ({var_explained_corr[1]:.1f}%)')
        plt.grid(True, alpha=0.3)
        
        # Annotate reduced batch effect
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        plt.annotate('Reduced batch effect', xy=(x_min + (x_max-x_min)*0.05, y_min + (y_max-y_min)*0.05), 
                     xycoords='data', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.3))
    else:
        # Plot only original data PCA
        plt.figure(figsize=(10, 6))
        if batch_info_raw is not None:
            batches = batch_info_raw.unique()
            cmap = plt.cm.get_cmap('viridis', len(batches))
            
            for i, batch in enumerate(batches):
                batch_indices = batch_info_raw == batch
                plt.scatter(pca_raw[batch_indices, 0], pca_raw[batch_indices, 1], 
                           c=[cmap(i)], label=f'Batch {batch}', alpha=0.7, edgecolors='none')
            plt.legend(title='Batch')
        else:
            plt.scatter(pca_raw[:, 0], pca_raw[:, 1], c='blue', alpha=0.7, edgecolors='none')
            
        plt.title('PCA of Original Data (Batch Effect Visible)', fontweight='bold')
        plt.xlabel(f'Principal Component 1 ({var_explained_raw[0]:.1f}%)')
        plt.ylabel(f'Principal Component 2 ({var_explained_raw[1]:.1f}%)')
        plt.grid(True, alpha=0.3)
        
        # Add visual annotations if needed
        # x_min, x_max = plt.xlim()
        # y_min, y_max = plt.ylim()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Fig1_BatchEffect_PCA.png')
    plt.close()
    print(f'Saved {output_dir}/Fig1_BatchEffect_PCA.png')
    
except Exception as e:
    print(f"[PCA Batch Effect] Error: {e}")

# 2. SMOTE前后类别分布对比 (Class balance comparison before/after SMOTE)
try:
    df_raw = pd.read_csv('four.csv')
    df_smote = pd.read_csv('logfourupsample.csv')
    print('Successfully loaded class distribution data')
    
    # Check column names for class labels
    raw_class_col = 'response' if 'response' in df_raw.columns else 'label'
    smote_class_col = 'response' if 'response' in df_smote.columns else 'label'
    
    plt.figure(figsize=(10, 5))
    
    # Plot original data class distribution
    plt.subplot(1, 2, 1)
    raw_counts = df_raw[raw_class_col].value_counts().sort_index()
    ax1 = raw_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Class Distribution - Original Data', fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    
    # Add count labels on bars
    for i, v in enumerate(raw_counts.values):
        ax1.text(i, v + 0.1, str(v), ha='center')
    
    # Plot SMOTE-upsampled data class distribution
    plt.subplot(1, 2, 2)
    smote_counts = df_smote[smote_class_col].value_counts().sort_index()
    ax2 = smote_counts.plot(kind='bar', color='orange', edgecolor='black')
    plt.title('Class Distribution - After SMOTE', fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    
    # Add count labels on bars
    for i, v in enumerate(smote_counts.values):
        ax2.text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Fig2_SMOTE_ClassBalance.png')
    plt.close()
    print(f'Saved {output_dir}/Fig2_SMOTE_ClassBalance.png')
    
except Exception as e:
    print(f"[SMOTE Class Balance] Error: {e}")

# 3. 模型架构示意图 (Model architecture diagram)
def draw_amu_architecture():
    try:
        from graphviz import Digraph
        print('Graphviz imported successfully')
        
        # Create a more detailed AMU architecture diagram
        dot = Digraph(comment='AMU Architecture', format='png')
        dot.attr(rankdir='TB', size='10,8', dpi='300')
        
        # Define node styles for different components
        dot.attr('node', shape='box', style='filled,rounded', color='lightblue', fontname='Arial', fontsize='12')
        
        # Input & Processing Nodes
        dot.node('A', 'Input Layer\n(mRNA Expression Profiles)\n160 Genes', style='filled', fillcolor='#E6F3FF')
        dot.node('B', 'Embedding Layer\n(Feature Transformation)\nDim: 20', style='filled', fillcolor='#D1E7FF')
        
        # Attention Mechanism
        with dot.subgraph(name='cluster_attention') as c:
            c.attr(label='Attention Module', style='filled,rounded', fillcolor='#F0F5FF')
            c.node('C1', 'Multi-Head Self-Attention\n(10 Heads)', style='filled', fillcolor='#BBDBFF')
            c.node('C2', 'Transformer Encoder\n(8 Layers)', style='filled', fillcolor='#BBDBFF')
        
        # Feature Processing
        dot.node('D', 'Feed-Forward Network\n(Dim: 200)', style='filled', fillcolor='#A6CFFF')
        dot.node('E', 'Layer Normalization', style='filled', fillcolor='#8FC2FF')
        dot.node('F1', 'Conv1D Layer\n(5 Filters)', style='filled', fillcolor='#79B6FF')
        dot.node('F2', 'Max Pooling', style='filled', fillcolor='#63AAFF')
        dot.node('G', 'Fully Connected Layer\n(Feature Compression)', style='filled', fillcolor='#4D9EFF')
        dot.node('H', 'Output Layer\n(Binary Classification)', style='filled', fillcolor='#3792FF')
        
        # Define edges with labels
        dot.edge('A', 'B', label='Gene Embedding')
        dot.edge('B', 'C1')
        dot.edge('C1', 'C2', label='Feature Interactions')
        dot.edge('C2', 'D')
        dot.edge('D', 'E', label='Non-linear Transformation')
        dot.edge('E', 'F1')
        dot.edge('F1', 'F2')
        dot.edge('F2', 'G', label='Dimensionality Reduction')
        dot.edge('G', 'H', label='Classification')
        
        # Add a caption
        dot.attr(label='\n\nFigure 3: Detailed Architecture of Attention-based Multi-layer Unit (AMU) Model\n' +
                 'The model leverages transformer encoders and multi-head attention mechanisms to identify complex gene interactions', 
                 fontsize='14', fontname='Arial')
        
        # Render the graph
        dot.render(f'{output_dir}/Fig3_AMU_Architecture', cleanup=True)
        print(f'Saved {output_dir}/Fig3_AMU_Architecture.png')
        
    except ImportError:
        print('Warning: Graphviz not installed or not in PATH. Using matplotlib fallback visualization...')
        
        # Generate a more visual diagram using matplotlib as fallback
        try:
            plt.figure(figsize=(12, 10), dpi=150)
            plt.axis('off')
            
            # Define layer positions and sizes
            layer_positions = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
            box_height = 0.08
            layer_colors = ['#E6F3FF', '#D1E7FF', '#BBDBFF', '#A6CFFF', '#8FC2FF', '#63AAFF', '#3792FF']
            layer_labels = [
                'Input Layer\n(mRNA Expression Profiles - 160 Genes)',
                'Embedding Layer\n(Feature Transformation - Dim: 20)',
                'Multi-Head Self-Attention\n(10 Heads, 8 Encoder Layers)',
                'Feed-Forward Network\n(Dim: 200)',
                'Layer Normalization +\nConv1D (5 Filters) + Pooling',
                'Fully Connected Layer\n(Feature Compression)',
                'Output Layer\n(Binary Classification)'
            ]
            
            # Draw boxes and labels
            for i, (pos, label, color) in enumerate(zip(layer_positions, layer_labels, layer_colors)):
                width = 0.7 if i == 2 else 0.5  # Make attention layer wider
                rect = plt.Rectangle((0.5-width/2, pos), width, box_height, 
                                     facecolor=color, edgecolor='black', alpha=0.8, 
                                     linewidth=1.5, zorder=2)
                plt.gca().add_patch(rect)
                plt.text(0.5, pos+box_height/2, label, ha='center', va='center', 
                         fontsize=12, fontweight='bold', zorder=3)
                
                # Add arrows between layers
                if i < len(layer_positions) - 1:
                    plt.arrow(0.5, pos+box_height+0.01, 0, layer_positions[i+1]-pos-box_height-0.02,
                              head_width=0.02, head_length=0.02, fc='black', ec='black', zorder=1)
            
            # Add model name and description
            plt.text(0.5, 0.02, 'Figure 3: Attention-based Multi-layer Unit (AMU) Model Architecture', 
                     ha='center', fontsize=14, fontweight='bold')
            
            # Add detailed description
            description = (
                "AMU leverages transformer encoders and multi-head attention mechanisms\n"
                "to model complex interactions between gene expressions for melanoma immunotherapy response prediction.\n"
                "The self-attention mechanism allows the model to identify key gene interactions\n"
                "that contribute to treatment response by assigning attention weights to different genes."
            )
            plt.text(0.5, 0.96, description, ha='center', fontsize=11, 
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/Fig3_AMU_Architecture.png')
            plt.close()
            print(f'Saved enhanced fallback architecture diagram: {output_dir}/Fig3_AMU_Architecture.png')
        except Exception as e:
            print(f"[Enhanced Fallback Architecture Diagram] Error: {e}")
    
    except Exception as e:
        print(f"[AMU Architecture] Error: {e}")

draw_amu_architecture()

# 4. 模型性能对比图 (Model performance comparison)
try:
    # Load model metrics data - note that model names are in the index, not as a column
    model_metrics = pd.read_csv('model_comparison.csv')
    print('Successfully loaded model_comparison.csv')
    
    # Reset index to make the model names a column
    model_metrics = model_metrics.reset_index()
    model_metrics.rename(columns={'index': 'Model'}, inplace=True)
    
    # Plot bar chart for each metric across models
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AP']
    models = model_metrics['Model'].values
    
    # Prepare a more professional-looking plot
    plt.figure(figsize=(14, 8))
    
    # Use a custom color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Set width of bars
    bar_width = 0.15
    positions = np.arange(len(models))
    
    # Plot each metric as a group of bars
    for i, metric in enumerate(metrics):
        plt.bar(positions + i*bar_width, model_metrics[metric], 
                width=bar_width, label=metric, color=colors[i % len(colors)])
    
    # Add labels and title
    plt.xlabel('Models', fontsize=14, fontweight='bold')
    plt.ylabel('Performance Score', fontsize=14, fontweight='bold')
    plt.title('Comparison of Model Performance Metrics', fontsize=16, fontweight='bold')
    plt.xticks(positions + bar_width * (len(metrics) - 1) / 2, models, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            value = model_metrics.loc[model_metrics['Model'] == model, metric].values[0]
            plt.text(j + i*bar_width, value + 0.02, f'{value:.2f}', 
                     ha='center', va='bottom', fontsize=9, rotation=90)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Fig4_Model_Performance_Comparison.png')
    plt.close()
    print(f'Saved {output_dir}/Fig4_Model_Performance_Comparison.png')
    
    # Create a separate heatmap for model comparison using a simpler approach
    try:
        plt.figure(figsize=(10, 6))
        # Create data for the heatmap manually to avoid index issues
        model_names = model_metrics['Model'].tolist()
        metrics_data = model_metrics[metrics].values
        
        # Create heatmap with a simpler method
        hm = plt.imshow(metrics_data, cmap='YlGnBu', aspect='auto')
        plt.colorbar(hm)
        
        # Add annotations
        for i in range(len(model_names)):
            for j in range(len(metrics)):
                value = metrics_data[i, j]
                plt.text(j, i, f'{value:.3f}', ha='center', va='center', 
                         color='black' if value > 0.7 else 'white', fontweight='bold')
        
        # Add axis labels
        plt.yticks(range(len(model_names)), model_names)
        plt.xticks(range(len(metrics)), metrics)
        plt.title('Model Performance Metrics Heatmap', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Fig5_Model_Performance_Heatmap.png')
        plt.close()
        print(f'Saved {output_dir}/Fig5_Model_Performance_Heatmap.png')
    except Exception as e:
        print(f"[Model Performance Heatmap] Error: {e}")
    
except Exception as e:
    print(f"[Model Performance Comparison] Error: {e}")

# 5. 基因表达热图 (Gene expression heatmap)
try:
    # Load gene expression data
    gene_data = pd.read_csv('four.csv', index_col=0)
    print('Preparing gene expression heatmap...')
    
    # Get class labels if available
    has_response = 'response' in gene_data.columns
    
    if has_response:
        # Separate features and response
        X = gene_data.drop('response', axis=1)
        y = gene_data['response']
        
        # Select top genes by variance for visualization (to avoid overcrowded heatmap)
        gene_variance = X.var().sort_values(ascending=False)
        top_genes = gene_variance.head(25).index.tolist()
        X_top = X[top_genes]
        
        # Create a heatmap with class labels
        plt.figure(figsize=(14, 10))
        
        # Sort samples by class for better visualization
        sample_order = gene_data.sort_values('response').index
        
        # Plot heatmap with samples sorted by class
        sns.heatmap(X_top.loc[sample_order].T, cmap='viridis', 
                    yticklabels=True, xticklabels=False,
                    cbar_kws={'label': 'Expression Level'})
        
        # Add color bar for classes on top
        ax2 = plt.gca()
        ax2.set_title('Top 25 Genes by Variance - Expression Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Samples (sorted by class)', fontsize=14)
        plt.ylabel('Genes', fontsize=14)
        
        # Create class indicator bar on top
        class_colors = {0: 'blue', 1: 'red'}
        class_bar = pd.DataFrame({'Class': y.loc[sample_order]})
        
        # Add the class bar on top of the heatmap
        ax_top = plt.axes([0.1, 0.92, 0.8, 0.02], frameon=True)
        for i, cls in enumerate(sorted(y.unique())):
            mask = class_bar['Class'] == cls
            ax_top.barh(0, width=mask.sum(), left=mask.cumsum().shift(fill_value=0), 
                        height=1, color=class_colors.get(cls, f'C{i}'), 
                        label=f'Class {cls}')
        
        ax_top.set_ylim(0, 1)
        ax_top.set_xlim(0, len(sample_order))
        ax_top.set_yticks([])
        ax_top.set_xticks([])
        ax_top.legend(loc='upper center', bbox_to_anchor=(0.5, 3), 
                     ncol=len(y.unique()), frameon=True)
    else:
        # If no response column, just create a simple gene expression heatmap
        gene_variance = gene_data.var().sort_values(ascending=False)
        top_genes = gene_variance.head(25).index.tolist()
        X_top = gene_data[top_genes]
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(X_top.T, cmap='viridis', yticklabels=True, xticklabels=False,
                   cbar_kws={'label': 'Expression Level'})
        plt.title('Top 25 Genes by Variance - Expression Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Samples', fontsize=14)
        plt.ylabel('Genes', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Fig6_Gene_Expression_Heatmap.png')
    plt.close()
    print(f'Saved {output_dir}/Fig6_Gene_Expression_Heatmap.png')
    
except Exception as e:
    print(f"[Gene Expression Heatmap] Error: {e}")

# 6. 数据预处理流程图 (Data preprocessing flowchart)
try:
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    
    # Create flowchart as a text-based diagram
    preprocessing_text = (
        "Data Preprocessing Workflow\n\n"
        "Raw Gene Expression Data\n↓\n"
        "Quality Control & Filtering\n↓\n"
        "Log Transformation\n↓\n"
        "Batch Effect Correction\n↓\n"
        "SMOTE Class Balancing\n↓\n"
        "Feature Selection/Dimensionality Reduction\n↓\n"
        "Data Split (Train/Validation/Test)\n↓\n"
        "Model Training & Evaluation"
    )
    
    # Draw boxes and arrows
    plt.text(0.5, 0.5, preprocessing_text, ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Fig7_Data_Preprocessing_Flowchart.png')
    plt.close()
    print(f'Saved {output_dir}/Fig7_Data_Preprocessing_Flowchart.png')
    
except Exception as e:
    print(f"[Data Preprocessing Flowchart] Error: {e}")

print('All figure scripts executed. Check the figures/ directory for generated images.')
print('If any figure is missing, please check the error messages above.')
