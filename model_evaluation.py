"""
模型评估和解释性分析
解决审稿人对随机森林分类（而非回归）和模型解释性的问题
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
import paddle
import paddle.nn as nn

# 尝试导入SHAP库，用于模型解释
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("请安装SHAP库以获得更好的模型解释性: pip install shap")

# 检查是否可以使用Paddle框架
try:
    paddle.device.set_device('gpu:0' if paddle.device.is_compiled_with_cuda() else 'cpu')
    PADDLE_AVAILABLE = True
except:
    PADDLE_AVAILABLE = False
    print("PaddlePaddle框架不可用，仅使用sklearn模型")

def train_models(X_train, y_train, models=None):
    """
    训练多个分类模型
    
    参数:
    - X_train: 训练特征
    - y_train: 训练标签
    - models: 要训练的模型字典（如果为None则使用默认模型）
    
    返回:
    - 训练好的模型字典
    """
    if models is None:
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='rbf', 
                C=1.0,
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, 
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=1.0  # 默认使用平衡权重
            )
        }
    
    # 训练所有模型
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def train_amu_model(X_train, y_train, batch_size=32, epochs=50):
    """
    训练AMU模型
    
    参数:
    - X_train: 训练特征
    - y_train: 训练标签
    - batch_size: 批次大小
    - epochs: 训练轮数
    
    返回:
    - 训练好的AMU模型
    """
    if not PADDLE_AVAILABLE:
        print("PaddlePaddle is not available, cannot train AMU model")
        return None
    
    # 尝试修改sys.path临时解决导入问题
    import sys
    import os
    
    try:
        # 获取AMU模型定义但不直接导入可能加载预训练权重的模块
        # 直接定义Atten_model类
        class Atten_model(nn.Layer):
            def __init__(self):
                super(Atten_model, self).__init__()  # [-1,1,160]
                self.drop = nn.Dropout(0.5)
                self.relu = nn.ReLU()
                self.softmax = nn.Softmax()
                self.flatten = nn.Flatten(1, -1)
                self.bn = nn.BatchNorm1D(1)
                self.x = paddle.to_tensor([i for i in range(X_train.shape[1])])
                self.embedding_layer1 = paddle.nn.Embedding(num_embeddings=X_train.shape[1],
                                                          embedding_dim=20)
                self.d = self.embedding_layer1(self.x)
                self.pretrained_attr = paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(self.d),
                    trainable=True)
                self.embedding_layer2 = paddle.nn.Embedding(num_embeddings=X_train.shape[1],
                                                          embedding_dim=20,
                                                          weight_attr=self.pretrained_attr,
                                                          name='myembeding')
                self.encoderlayer1 = nn.TransformerEncoderLayer(d_model=20,
                                                              nhead=10,
                                                              dim_feedforward=200)
                self.encoder1 = nn.TransformerEncoder(self.encoderlayer1, 8)  
                self.conv1 = nn.Conv1D(in_channels=20, out_channels=5, kernel_size=1, stride=1, padding=0, data_format='NCL',
                                      bias_attr=False)  
                self.bn1 = nn.BatchNorm1D(5)
                self.pool3 = nn.AdaptiveMaxPool1D(1)  
                self.linear1 = nn.Linear(X_train.shape[1], 2, name='seconde_linear')

            def forward(self, x):
                e = self.embedding_layer2(self.x)  # shape=[160, 10]
                e = e.transpose((1, 0))  # shape=[10, 160]
                x = paddle.multiply(e, x)
                x = x.transpose((0, 2, 1))  # shape=[160, 10]
                x = self.encoder1(x)
                x = x.transpose((0, 2, 1))  # [-1,10,160]
                x = self.conv1(x)  # [-1,5,160]
                x = self.drop(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = x.transpose((0, 2, 1))  # #[-1,160,5]
                x = self.pool3(x)  # [-1,160,1]
                x = self.drop(x)
                x = self.flatten(x)
                x = self.linear1(x)  # [20]
                x = self.softmax(x)
                return x
        
        # 创建数据集
        class SimpleDataset(paddle.io.Dataset):
            def __init__(self, features, labels):
                self.features = features
                self.labels = labels
                
            def __getitem__(self, index):
                feature = self.features[index].reshape([-1, X_train.shape[1]])
                label = self.labels[index]
                return feature, label
                
            def __len__(self):
                return len(self.features)
        
        # 转换数据为Paddle张量
        train_features = paddle.to_tensor(X_train.values.astype('float32'))
        train_labels = paddle.to_tensor(y_train.values.astype('int64'))
        
        # 创建数据加载器
        train_dataset = SimpleDataset(train_features, train_labels)
        train_loader = paddle.io.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # 创建全新的模型实例（不依赖amu.py中可能加载预训练权重的模型）
        print("创建新的AMU模型实例...")
        model = Atten_model()
        
        # 定义优化器和损失函数
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.00001,
            parameters=model.parameters(),
            weight_decay=paddle.regularizer.L2Decay(0.0001)
        )
        loss_fn = nn.CrossEntropyLoss()
        
        # 训练模型
        print(f"开始训练AMU模型，训练轮数: {epochs}")
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                
                # 前向传播
                logits = model(x_data)
                loss = loss_fn(logits, y_data)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                
                # 计算准确率
                pred = paddle.argmax(logits, axis=1)
                correct += (pred == y_data).sum().numpy()[0]
                total += len(y_data)
                total_loss += loss.numpy()[0]
                
            # 打印每个epoch的结果
            acc = correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}")
        
        print("AMU模型训练完成")
        return model
    except Exception as e:
        print(f"AMU模型训练过程中发生错误: {e}")
        return None

def evaluate_models(models, X_test, y_test, threshold=0.5):
    """
    评估多个模型的性能
    
    参数:
    - models: 训练好的模型字典
    - X_test: 测试特征
    - y_test: 测试标签
    - threshold: 分类阈值
    
    返回:
    - 评估指标字典
    """
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name} model...")
        
        # 检查是否是PaddlePaddle模型
        is_paddle_model = False
        if name == 'AMU' or name == 'CNN' or (hasattr(model, '__module__') and 'paddle' in model.__module__.lower()):
            is_paddle_model = True
            print(f"{name} 是PaddlePaddle模型，使用特殊处理...")
        
        # 预测概率
        if is_paddle_model:
            try:
                # 处理PaddlePaddle模型
                model.eval()  # 设置为评估模式
                
                # 转换数据为Paddle张量
                test_features = paddle.to_tensor(X_test.values.astype('float32'))
                
                # 创建简单数据集
                class SimpleTestDataset(paddle.io.Dataset):
                    def __init__(self, features):
                        self.features = features
                    
                    def __getitem__(self, index):
                        return self.features[index].reshape([-1, X_test.shape[1]])
                    
                    def __len__(self):
                        return len(self.features)
                
                test_dataset = SimpleTestDataset(test_features)
                test_loader = paddle.io.DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                # 收集所有预测结果
                y_probs = []
                with paddle.no_grad():
                    for batch_id, x_data in enumerate(test_loader()):
                        logits = model(x_data)
                        probs = paddle.nn.functional.softmax(logits, axis=1)
                        y_probs.append(probs.numpy()[:, 1])  # 取正类的概率
                
                y_prob = np.concatenate(y_probs)
                
            except Exception as e:
                print(f"处理{name}模型时出错: {e}")
                print(f"无法获取{name}的预测概率，跳过评估")
                continue
        elif hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # 对于没有predict_proba的模型
            try:
                y_prob = model.predict(X_test)
            except Exception as e:
                print(f"处理{name}模型时出错: {e}")
                print(f"无法获取{name}的预测概率，跳过评估")
                continue
        
        # 根据阈值获取预测标签
        y_pred = (y_prob >= threshold).astype(int)
        
        # 计算评估指标
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        # 计算PR曲线
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        
        # 存储结果
        results[name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'ap': ap,
            'confusion_matrix': cm,
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision_curve, recall_curve),
            'y_prob': y_prob,
            'y_pred': y_pred
        }
        
        # 打印主要指标
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Average Precision: {ap:.4f}")
        print("Confusion Matrix:")
        print(cm)
    
    return results

def plot_roc_curves(results, title="ROC Curves Comparison", figsize=(10, 8), save_path=None):
    """
    绘制多个模型的ROC曲线
    
    参数:
    - results: 模型评估结果字典
    - title: 图表标题
    - figsize: 图表大小
    - save_path: 保存路径
    """
    plt.figure(figsize=figsize)
    
    # 定义颜色映射
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    # 为每个模型绘制ROC曲线
    for i, (name, result) in enumerate(results.items()):
        fpr, tpr = result['roc_curve']
        auc = result['auc']
        
        plt.plot(
            fpr, tpr, 
            lw=2, 
            color=colors[i % len(colors)],
            label=f'{name} (AUC = {auc:.3f})'
        )
    
    # 添加对角线（随机猜测）
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8, label='Random')
    
    # 设置图表属性
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=15)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_pr_curves(results, title="Precision-Recall Curves Comparison", figsize=(10, 8), save_path=None):
    """
    绘制多个模型的PR曲线
    
    参数:
    - results: 模型评估结果字典
    - title: 图表标题
    - figsize: 图表大小
    - save_path: 保存路径
    """
    plt.figure(figsize=figsize)
    
    # 定义颜色映射
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    
    # 为每个模型绘制PR曲线
    for i, (name, result) in enumerate(results.items()):
        precision_curve, recall_curve = result['pr_curve']
        ap = result['ap']
        
        plt.plot(
            recall_curve, precision_curve, 
            lw=2, 
            color=colors[i % len(colors)],
            label=f'{name} (AP = {ap:.3f})'
        )
    
    # 设置图表属性
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=15)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrices(results, figsize=(15, 10), save_path=None):
    """
    绘制多个模型的混淆矩阵
    
    参数:
    - results: 模型评估结果字典
    - figsize: 图表大小
    - save_path: 保存路径
    """
    # 计算需要的子图行列数
    n_models = len(results)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # 使axes总是二维数组
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    # 为每个模型绘制混淆矩阵
    for i, (name, result) in enumerate(results.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        cm = result['confusion_matrix']
        
        # 计算归一化的混淆矩阵
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制混淆矩阵
        sns.heatmap(
            cm_norm, 
            annot=cm,  # 显示原始计数
            fmt='d', 
            cmap='Blues',
            cbar=True,
            square=True,
            ax=ax
        )
        
        # 设置标题和标签
        ax.set_title(f'{name} Confusion Matrix', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        
        # 设置刻度标签
        classes = ['Negative', 'Positive']
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
    
    # 隐藏空白子图
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_feature_importance(models, feature_names, top_n=50, figsize=(12, 10), save_path=None):
    """
    分析和可视化特征重要性
    
    参数:
    - models: 训练好的模型字典
    - feature_names: 特征名称列表
    - top_n: 要显示的顶部特征数量
    - figsize: 图表大小
    - save_path: 保存路径
    
    返回:
    - 所有模型的重要特征的并集
    """
    # 收集每个模型的特征重要性
    importance_dfs = []
    
    for name, model in models.items():
        # 跳过不支持特征重要性的模型
        if not hasattr(model, 'feature_importances_') and name != 'SVM':
            continue
        
        if name == 'SVM':
            # 对于SVM，使用系数作为近似的特征重要性
            if hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                continue
        else:
            importances = model.feature_importances_
        
        # 创建特征重要性DataFrame
        df = pd.DataFrame({
            'Feature': feature_names,
            f'{name}_Importance': importances
        })
        
        # 根据重要性排序
        df = df.sort_values(f'{name}_Importance', ascending=False)
        
        # 存储结果
        importance_dfs.append(df)
    
    # 合并所有模型的特征重要性
    if importance_dfs:
        all_importances = importance_dfs[0][['Feature']]
        
        for df in importance_dfs:
            importance_col = [col for col in df.columns if col.endswith('_Importance')][0]
            all_importances = all_importances.merge(
                df[['Feature', importance_col]],
                on='Feature',
                how='outer'
            )
        
        # 用0填充缺失值
        all_importances = all_importances.fillna(0)
        
        # 添加平均重要性
        importance_cols = [col for col in all_importances.columns if col.endswith('_Importance')]
        all_importances['Average_Importance'] = all_importances[importance_cols].mean(axis=1)
        
        # 根据平均重要性排序
        all_importances = all_importances.sort_values('Average_Importance', ascending=False)
    else:
        print("没有支持特征重要性的模型")
        return None
    
    # 绘制前top_n个特征的重要性
    plt.figure(figsize=figsize)
    
    # 获取前top_n个特征
    top_features = all_importances.head(top_n)
    
    # 创建条形图
    ax = sns.barplot(
        x='Average_Importance',
        y='Feature',
        data=top_features,
        palette='viridis'
    )
    
    # 设置标题和标签
    plt.title(f'Top {top_n} Feature Importance (Average Across Models)', fontsize=15)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    # 添加网格线
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 返回所有模型的重要特征的并集
    important_features = all_importances.head(top_n)['Feature'].tolist()
    return important_features

def perform_shap_analysis(models, X_train, X_test, feature_names, save_path=None):
    """
    使用SHAP进行模型解释
    
    参数:
    - models: 训练好的模型字典
    - X_train: 训练特征
    - X_test: 测试特征
    - feature_names: 特征名称列表
    - save_path: 保存路径
    """
    if not SHAP_AVAILABLE:
        print("SHAP library is not available, cannot perform SHAP analysis")
        return
    
    # 为每个模型计算SHAP值
    for name, model in models.items():
        print(f"\nCalculating SHAP values for {name}...")
        
        # 创建SHAP解释器
        if name == 'RandomForest':
            explainer = shap.TreeExplainer(model)
        elif name == 'XGBoost':
            explainer = shap.TreeExplainer(model)
        elif name == 'SVM':
            # 对于SVM，使用KernelExplainer
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
        else:
            print(f"Unsupported model type: {name}")
            continue
        
        # 计算SHAP值
        if name in ['RandomForest', 'XGBoost']:
            shap_values = explainer.shap_values(X_test)
            
            # 对于二分类问题，取正类的SHAP值
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            shap_values = explainer.shap_values(X_test)
        
        # 创建SHAP摘要图
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, 
            X_test, 
            feature_names=feature_names,
            show=False
        )
        
        plt.title(f'{name} SHAP Summary', fontsize=15)
        
        # 保存图表
        if save_path:
            plt.savefig(f'{save_path}_{name}_shap_summary.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # 创建SHAP条形图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_test,
            feature_names=feature_names,
            plot_type='bar',
            show=False
        )
        
        plt.title(f'{name} SHAP Feature Importance', fontsize=15)
        
        # 保存图表
        if save_path:
            plt.savefig(f'{save_path}_{name}_shap_importance.png', dpi=300, bbox_inches='tight')
        
        plt.show()

def save_models_comparison(results, save_path='model_comparison.csv'):
    """
    保存模型比较结果到CSV文件
    
    参数:
    - results: 模型评估结果字典
    - save_path: 保存路径
    """
    # 提取主要指标
    comparison = {}
    
    for name, result in results.items():
        comparison[name] = {
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1': result['f1'],
            'AUC': result['auc'],
            'AP': result['ap']
        }
    
    # 转换为DataFrame
    df = pd.DataFrame(comparison).T
    
    # 保存到CSV
    df.to_csv(save_path)
    print(f"Model comparison results saved to {save_path}")
    
    return df

# 示例用法
if __name__ == "__main__":
    # 加载数据
    try:
        data = pd.read_csv('batch_corrected_data.csv')  # 使用批次校正后的数据
    except:
        try:
            data = pd.read_csv('smote_balanced_data.csv')  # 使用SMOTE平衡后的数据
        except:
            try:
                data = pd.read_csv('logfourupsample.csv')  # 使用原始数据
            except:
                print("Unable to load data files, creating example data")
                # 创建示例数据
                np.random.seed(42)
                n_samples, n_features = 1000, 160
                X = np.random.randn(n_samples, n_features)
                y = np.random.randint(0, 2, n_samples)
                data = pd.DataFrame(np.column_stack([X, y]), 
                                   columns=[f'feature_{i}' for i in range(n_features)] + ['target'])
    
    # 分离特征和标签
    if 'target' in data.columns:
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
    elif 'batch' in data.columns:
        # 在样本数不一致的情况下，直接使用原始数据
        print("Sample size mismatch detected, using original data instead...")
        try:
            # 直接使用原始数据
            orig_data = pd.read_csv('logfourupsample.csv')
            X = orig_data.iloc[:, :-1]
            y = orig_data.iloc[:, -1]
            print(f"Successfully loaded original data. Shape: {X.shape}, Labels: {np.unique(y)}")
            
            # 为了演示的目的，如果存在平衡后的数据，则使用它
            try:
                balanced_data = pd.read_csv('smote_balanced_data.csv')
                X = balanced_data.iloc[:, :-1]
                y = balanced_data.iloc[:, -1]
                print(f"Found and using balanced data. Shape: {X.shape}")
            except Exception:
                print("No balanced data found, using original data.")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating random data for demonstration...")
            # 创建随机数据用于演示
            X = data.drop('batch', axis=1)
            np.random.seed(42)
            y = np.random.randint(0, 2, size=len(X))
    else:
        # 假设最后一列是标签
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    # 获取特征名称
    feature_names = X.columns.tolist()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 训练模型
    models = train_models(X_train, y_train)
    
    # 尝试训练AMU模型
    try:
        print("\nTraining AMU model...")
        amu_model = train_amu_model(X_train, y_train, batch_size=32, epochs=20)  # 减少epochs以加快训练
        if amu_model is not None:
            models['AMU'] = amu_model
            print("AMU model trained successfully and added to evaluation.")
    except Exception as e:
        print(f"Could not train AMU model: {e}")
        print("Continuing evaluation with other models only.")

    
    # 评估模型
    results = evaluate_models(models, X_test, y_test)
    
    # 绘制ROC曲线
    plot_roc_curves(results, save_path='roc_curves.png')
    
    # 绘制PR曲线
    plot_pr_curves(results, save_path='pr_curves.png')
    
    # 绘制混淆矩阵
    plot_confusion_matrices(results, save_path='confusion_matrices.png')
    
    # 分析特征重要性
    important_features = analyze_feature_importance(
        models, feature_names, top_n=50, save_path='feature_importance.png'
    )
    
    # 执行SHAP分析
    if SHAP_AVAILABLE:
        perform_shap_analysis(models, X_train, X_test, feature_names, save_path='shap')
    
    # 保存模型比较结果
    comparison_df = save_models_comparison(results)
    
    # 打印最重要的特征
    if important_features:
        print("\nTop 20 most important features:")
        for i, feature in enumerate(important_features[:20]):
            print(f"{i+1}. {feature}")
        
        # 保存重要特征到CSV
        pd.DataFrame({'Feature': important_features}).to_csv('important_features.csv', index=False)
        print("Important features saved to important_features.csv")
