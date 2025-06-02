"""
多模型评估脚本 - 使用相同的数据加载方式评估多个模型
包括AMU模型和CNN模型
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as ms_train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            roc_curve, precision_recall_curve, auc)
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.regularizer import L2Decay
from paddle.io import Dataset, DataLoader
import time
import os

# 创建结果目录
os.makedirs('model_results', exist_ok=True)

# ------------ 按照用户提供的代码实现数据加载 ------------
print("开始加载数据...")
try:
    data = pd.read_csv('logfourupsample.csv', sep=',')
    print(f"成功加载数据，形状: {data.shape}")
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit(1)

# 整理数据集，拆分测试集训练集，使用原始的随机种子1000
x, y = data.iloc[:, :-1], data.iloc[:, -1]
train_x, test_x, train_y, test_y = ms_train_test_split(x, y, test_size=0.2, random_state=1000)
print(f"训练集形状: {train_x.shape}, 测试集形状: {test_x.shape}")
print(f"标签分布 - 训练集: {train_y.value_counts().to_dict()}, 测试集: {test_y.value_counts().to_dict()}")

# 准备测试数据集
testdata = pd.concat([test_x, test_y], axis=1)
data_np = np.array(testdata).astype('float32')
selfdata = []
for i in range(data_np.shape[0]):
    input_np = data_np[i, :-1].reshape([-1, 160])
    label_np = data_np[i, -1].astype('int64')
    selfdata.append([input_np, label_np])
testdata = selfdata

# 为训练集准备类似的处理
train_data = pd.concat([train_x, train_y], axis=1)
train_np = np.array(train_data).astype('float32')
train_selfdata = []
for i in range(train_np.shape[0]):
    input_np = train_np[i, :-1].reshape([-1, 160])
    label_np = train_np[i, -1].astype('int64')
    train_selfdata.append([input_np, label_np])

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_list, mode='train'):
        super(CustomDataset, self).__init__()
        self.data = data_list
        self.mode = mode
    
    def __getitem__(self, index):
        data = self.data[index][0]
        label = self.data[index][1]
        return data, label
    
    def __len__(self):
        return len(self.data)

# 创建训练和测试数据集
train_dataset = CustomDataset(train_selfdata, mode='train')
test_dataset = CustomDataset(selfdata, mode='test')

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------ 模型定义 ------------
# 1. AMU模型定义
class Atten_model(nn.Layer):
    def __init__(self):
        super(Atten_model, self).__init__()  # [-1,1,160]
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten(1, -1)
        self.bn = nn.BatchNorm1D(1)
        self.x = paddle.to_tensor([i for i in range(160)])
        self.embedding_layer1 = paddle.nn.Embedding(num_embeddings=160,
                                                   embedding_dim=20)
        self.d = self.embedding_layer1(self.x)
        self.pretrained_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Assign(self.d),
            trainable=True)
        self.embedding_layer2 = paddle.nn.Embedding(num_embeddings=160,
                                                   embedding_dim=20,
                                                   weight_attr=self.pretrained_attr,
                                                   name='myembeding')
        self.encoderlayer1 = nn.TransformerEncoderLayer(d_model=20,
                                                       nhead=10,
                                                       dim_feedforward=200)
        self.encoder1 = nn.TransformerEncoder(self.encoderlayer1, 8)  # [-1,160,10] -->#[-1,10,160]
        self.conv1 = nn.Conv1D(in_channels=20, out_channels=5, kernel_size=1, stride=1, padding=0, data_format='NCL',
                              bias_attr=False)  # [-1,25,160]
        self.bn1 = nn.BatchNorm1D(5)
        self.pool3 = nn.AdaptiveMaxPool1D(1)  # [-1,160,1] -->#[160]

        self.linear1 = nn.Linear(160, 2, name='seconde_linear')
        # self.linear2=nn.Linear(10,2,name='seconde_linear')

    def embeding(self):
        embedding = nn.Embedding(num_embeddings=160, embedding_dim=10)
        data = paddle.randint(1, 2, shape=[160])
        embed = embedding(data)
        embed_tensor = paddle.to_tensor(embed)
        re = embed_tensor.transpose((1, 0))
        return re  # 50,160

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

# 2. CNN模型定义
class CNN_model(nn.Layer):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Conv1D(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1D(kernel_size=2)
        self.conv2 = nn.Conv1D(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1D(kernel_size=2)
        self.conv3 = nn.Conv1D(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1D(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 20, 128)  # 160 -> 80 -> 40 -> 20 (after 3 pooling layers)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        # CNN模型期望输入形状为 [batch_size, channels, seq_len]
        # 当前输入为 [batch_size, seq_len]，需要增加一个通道维度
        x = x.unsqueeze(1)  # [batch_size, 1, 160]
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, axis=1)
        return x

# 3. MLP模型定义
class MLP_model(nn.Layer):
    def __init__(self):
        super(MLP_model, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(160, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        x = F.softmax(x, axis=1)
        return x

# ------------ 训练与评估函数 ------------
def train_model(model, model_name, train_loader, test_loader, epochs=100, learning_rate=0.0001, weight_decay=0.0001):
    """
    训练模型并记录训练过程
    """
    print(f"\n开始训练 {model_name} 模型...")
    
    # 初始化优化器
    optimizer = paddle.optimizer.Adam(
        learning_rate=learning_rate,
        parameters=model.parameters(),
        weight_decay=L2Decay(weight_decay)
    )
    
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': []
    }
    
    # 训练循环
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_id, data in enumerate(train_loader):
            x_data, y_data = data
            
            # 前向传播
            logits = model(x_data)
            loss = loss_fn(logits, y_data)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            total_loss += float(loss)
            batch_count += 1
        
        # 计算平均损失
        avg_loss = total_loss / batch_count
        history['train_loss'].append(avg_loss)
        
        # 每10个epoch或最后一个epoch评估一次
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            # 评估训练集
            train_metrics = evaluate_model(model, train_loader)
            # 评估测试集
            test_metrics = evaluate_model(model, test_loader)
            
            # 记录指标
            history['train_acc'].append(train_metrics['accuracy'])
            history['test_acc'].append(test_metrics['accuracy'])
            history['train_precision'].append(train_metrics['precision'])
            history['train_recall'].append(train_metrics['recall'])
            history['train_f1'].append(train_metrics['f1'])
            history['test_precision'].append(test_metrics['precision'])
            history['test_recall'].append(test_metrics['recall'])
            history['test_f1'].append(test_metrics['f1'])
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            print(f"  训练集 - 准确率: {train_metrics['accuracy']:.4f}, 精确率: {train_metrics['precision']:.4f}, 召回率: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  测试集 - 准确率: {test_metrics['accuracy']:.4f}, 精确率: {test_metrics['precision']:.4f}, 召回率: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
            print(f"  训练集混淆矩阵:\n{train_metrics['confusion_matrix']}")
            print(f"  测试集混淆矩阵:\n{test_metrics['confusion_matrix']}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 训练时间
    training_time = time.time() - start_time
    print(f"{model_name} 模型训练完成! 训练时间: {training_time:.2f} 秒")
    
    # 最终评估
    final_train_metrics = evaluate_model(model, train_loader)
    final_test_metrics = evaluate_model(model, test_loader)
    
    # 计算ROC和PR曲线
    test_labels, test_preds, test_probs = get_predictions(model, test_loader)
    
    # 计算ROC曲线数据
    fpr, tpr, _ = roc_curve(test_labels, test_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # 计算PR曲线数据
    precision, recall, _ = precision_recall_curve(test_labels, test_probs[:, 1])
    pr_auc = auc(recall, precision)
    
    # 绘制并保存ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'model_results/{model_name}_roc_curve.png')
    
    # 绘制并保存PR曲线
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(f'model_results/{model_name}_pr_curve.png')
    
    # 绘制训练历史
    plt.figure(figsize=(12, 10))
    
    # 绘制损失
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率
    plt.subplot(2, 2, 2)
    plt.plot(range(0, epochs, 10), history['train_acc'], label='Training Accuracy')
    plt.plot(range(0, epochs, 10), history['test_acc'], label='Testing Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制精确率和召回率
    plt.subplot(2, 2, 3)
    plt.plot(range(0, epochs, 10), history['train_precision'], label='Training Precision')
    plt.plot(range(0, epochs, 10), history['test_precision'], label='Testing Precision')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(range(0, epochs, 10), history['train_recall'], label='Training Recall')
    plt.plot(range(0, epochs, 10), history['test_recall'], label='Testing Recall')
    plt.title('Model Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'model_results/{model_name}_training_history.png')
    
    # 保存模型
    try:
        paddle.save(model.state_dict(), f'model_results/{model_name}_params')
        paddle_model = paddle.Model(model)
        paddle_model.save(f'model_results/{model_name}')
        print(f"模型已保存为 model_results/{model_name}")
    except Exception as e:
        print(f"保存模型时出错: {e}")
    
    # 返回最终指标和训练历史
    final_metrics = {
        'train': final_train_metrics,
        'test': final_test_metrics,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'training_time': training_time
    }
    
    return final_metrics, history

def evaluate_model(model, data_loader):
    """
    评估模型性能
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with paddle.no_grad():
        for data in data_loader:
            x, y = data
            logits = model(x)
            probs = F.softmax(logits, axis=1)
            preds = paddle.argmax(probs, axis=1)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    
    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm
    }

def get_predictions(model, data_loader):
    """
    获取模型的预测结果、真实标签和预测概率
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with paddle.no_grad():
        for data in data_loader:
            x, y = data
            logits = model(x)
            probs = F.softmax(logits, axis=1)
            preds = paddle.argmax(probs, axis=1)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ------------ 主函数：训练和评估多个模型 ------------
def main():
    print("开始评估多个模型...")
    
    # 存储所有模型的结果
    all_results = {}
    
    # 定义模型列表
    models = {
        'AMU': Atten_model(),
        'CNN': CNN_model(),
        'MLP': MLP_model()
    }
    
    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"训练和评估 {model_name} 模型")
        print(f"{'='*50}")
        
        # 训练模型
        metrics, history = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=100,
            learning_rate=0.0001
        )
        
        # 存储结果
        all_results[model_name] = {
            'metrics': metrics,
            'history': history
        }
        
        # 输出模型性能
        print(f"\n{model_name} 模型最终性能:")
        print(f"训练集准确率: {metrics['train']['accuracy']:.4f}")
        print(f"测试集准确率: {metrics['test']['accuracy']:.4f}")
        print(f"测试集精确率: {metrics['test']['precision']:.4f}")
        print(f"测试集召回率: {metrics['test']['recall']:.4f}")
        print(f"测试集F1分数: {metrics['test']['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")
        print(f"训练时间: {metrics['training_time']:.2f} 秒")
    
    # 比较所有模型的性能
    print("\n所有模型性能比较:")
    print("="*80)
    print(f"{'模型名称':<10}{'训练准确率':<12}{'测试准确率':<12}{'精确率':<10}{'召回率':<10}{'F1分数':<10}{'ROC AUC':<10}{'PR AUC':<10}{'训练时间(秒)':<15}")
    print("-"*80)
    
    for model_name, result in all_results.items():
        metrics = result['metrics']
        print(f"{model_name:<10}{metrics['train']['accuracy']:.4f}{'      '}{metrics['test']['accuracy']:.4f}{'      '}{metrics['test']['precision']:.4f}{'    '}{metrics['test']['recall']:.4f}{'    '}{metrics['test']['f1']:.4f}{'    '}{metrics['roc_auc']:.4f}{'    '}{metrics['pr_auc']:.4f}{'    '}{metrics['training_time']:.2f}")
    
    print("="*80)
    
    # 绘制所有模型的ROC曲线比较
    plt.figure(figsize=(12, 10))
    
    for model_name, model in models.items():
        # 获取测试集预测
        test_labels, test_preds, test_probs = get_predictions(model, test_loader)
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(test_labels, test_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.savefig('model_results/all_models_roc_comparison.png')
    
    # 绘制所有模型的准确率比较
    plt.figure(figsize=(12, 6))
    
    model_names = list(all_results.keys())
    train_accs = [all_results[model]['metrics']['train']['accuracy'] for model in model_names]
    test_accs = [all_results[model]['metrics']['test']['accuracy'] for model in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, train_accs, width, label='Train Accuracy')
    plt.bar(x + width/2, test_accs, width, label='Test Accuracy')
    
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    plt.ylim([0, 1])
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(train_accs):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(test_accs):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.savefig('model_results/all_models_accuracy_comparison.png')
    
    print("\n所有图表和模型已保存到 model_results 目录")
    print("评估完成!")

if __name__ == "__main__":
    main()
