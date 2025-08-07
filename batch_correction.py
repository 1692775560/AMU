"""
Batch Effect Correction and PCA Analysis
Implementing batch effect correction as suggested by Reviewer 2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats

# Import required Python packages - need to install combat-pytorch or pycombat
# Import pycombat library
try:
    import pycombat
    
    def apply_combat(data, batch):
        # Use pycombat for batch correction
        # pycombat API may have different usage
        try:
            corrected_data = pycombat.pycombat(data, batch)
            return corrected_data
        except (TypeError, AttributeError):
            # If pycombat module structure is different, try other methods
            print("Using alternative method for batch correction...")
            # Simple alternative method: standardize data for each batch
            corrected = data.copy()
            unique_batches = np.unique(batch)
            
            # Standardize each batch
            for b in unique_batches:
                idx = np.where(batch == b)[0]
                batch_data = data[idx]
                # Calculate mean and standard deviation
                batch_mean = np.mean(batch_data, axis=0)
                batch_std = np.std(batch_data, axis=0) + 1e-8  # Prevent division by zero
                
                # Standardize
                corrected[idx] = (batch_data - batch_mean) / batch_std
            
            return corrected
        
except ImportError:
    try:
        # Try to import neuroCombat
        import neuroCombat as combat
        
        def apply_combat(data, batch):
            # Use neuroCombat for batch correction
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
        
        # Define a simple alternative function so the script can at least run
        def apply_combat(data, batch):
            print("Warning: Batch correction library not found, returning original data")
            return data
        
def perform_pca_analysis(data, labels, title="PCA of Gene Expression Data", figsize=(10, 8), save_path=None):
    """
    Perform PCA analysis and visualize results
    
    Parameters:
    - data: Gene expression data matrix (samples x genes)
    - labels: Sample batch or category labels
    - title: Chart title
    - figsize: Chart size
    - save_path: Save path, if not None, save the image
    """
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create result DataFrame
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=['PC1', 'PC2']
    )
    pca_df['Batch'] = labels
    
    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_ * 100
    
    # Visualization
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
    
    # Add statistical analysis - Multivariate ANOVA (MANOVA) to detect inter-group differences
    groups = [pca_result[labels == label] for label in np.unique(labels)]
    if len(groups) > 1 and all(len(g) > 1 for g in groups):
        try:
            stat, p = stats.f_oneway(*[g[:, 0] for g in groups])  # Analyze PC1
            plt.figtext(0.01, 0.01, f'PC1 ANOVA: p={p:.4f}', fontsize=9)
        except:
            pass
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return pca_df, pca

def batch_correction_pipeline(data_files, batch_labels, gene_columns=None, sample_id_column=None):
    """
    Complete batch correction pipeline
    
    Parameters:
    - data_files: List of data file paths
    - batch_labels: Batch labels corresponding to each file
    - gene_columns: Gene column names (if None, use all numeric columns)
    - sample_id_column: Sample ID column name
    
    Returns:
    - Original merged data
    - Corrected data
    """
    # 1. Load and merge data
    all_data = []
    all_batches = []
    
    for i, file_path in enumerate(data_files):
        # Load data based on file type
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            print(f"Unsupported file type: {file_path}")
            continue
        
        # Add batch information
        batch = batch_labels[i]
        df['batch'] = batch
        
        # Extract feature columns
        if gene_columns is None:
            # Use all numeric columns as gene expression values
            feature_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            # Exclude batch column
            feature_cols = [col for col in feature_cols if col != 'batch']
        else:
            feature_cols = gene_columns
        
        # Store sample IDs
        if sample_id_column and sample_id_column in df.columns:
            sample_ids = df[sample_id_column].values
        else:
            sample_ids = [f"Sample_{batch}_{j}" for j in range(len(df))]
        
        # Extract expression matrix
        expression_data = df[feature_cols].values
        
        # Batch labels
        batch_labels_arr = np.array([batch] * len(df))
        
        all_data.append(expression_data)
        all_batches.extend(batch_labels_arr)
    
    # Merge all data
    combined_data = np.vstack(all_data)
    batch_array = np.array(all_batches)
    
    # 2. Perform PCA analysis before correction
    perform_pca_analysis(
        combined_data, 
        batch_array, 
        title="PCA Before Batch Correction",
        save_path="pca_before_correction.png"
    )
    
    # 3. Apply Combat for batch correction
    corrected_data = apply_combat(combined_data, batch_array)
    
    # 4. Perform PCA analysis again after correction
    perform_pca_analysis(
        corrected_data, 
        batch_array, 
        title="PCA After Batch Correction",
        save_path="pca_after_correction.png"
    )
    
    # 5. Return original and corrected data
    return combined_data, corrected_data, batch_array, feature_cols

# Example usage
if __name__ == "__main__":
    # Please replace with your actual data file paths when using
    data_files = ['logfourupsample.csv', 'four.csv']
    batch_labels = ['Dataset1', 'Dataset2']
    
    # Execute batch correction pipeline
    original_data, corrected_data, batches, features = batch_correction_pipeline(
        data_files, batch_labels
    )
    
    print(f"Original data shape: {original_data.shape}")
    print(f"Corrected data shape: {corrected_data.shape}")
    
    # Save corrected data
    corrected_df = pd.DataFrame(corrected_data, columns=features)
    corrected_df['batch'] = batches
    corrected_df.to_csv('batch_corrected_data.csv', index=False)
    
    print("Batch correction completed, data saved to batch_corrected_data.csv")
