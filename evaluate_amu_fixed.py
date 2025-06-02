"""
AMU模型单独评估脚本 - 修正版
使用用户提供的确切AMU模型结构
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            precision_recall_curve, roc_curve, average_precision_score)
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.regularizer import L2Decay

# 设置随机种子以确保结果可重现
np.random.seed(42)
paddle.seed(42)

print("开始加载数据...")
# 加载数据
try:
    data = pd.read_csv('logfourupsample.csv')
    print(f"成功加载数据，形状: {data.shape}")
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit(1)

# 分离特征和标签
if 'target' in data.columns:
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
else:
    # 假设最后一列是标签
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

print(f"特征形状: {X.shape}, 标签形状: {y.shape}")
print(f"标签分布: {y.value_counts().to_dict()}")

# 确保特征数量为160，否则调整模型
num_features = X.shape[1]
print(f"特征数量: {num_features}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

# 定义AMU模型 - 使用用户提供的确切代码
class Atten_model(nn.Layer):
    def __init__(self, num_features=160):
        super(Atten_model, self).__init__()  # [-1,1,160]
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten(1, -1)
        self.bn = nn.BatchNorm1D(1)
        self.x = paddle.to_tensor([i for i in range(num_features)])
        self.embedding_layer1 = paddle.nn.Embedding(num_embeddings=num_features,
                                                    embedding_dim=20)
        self.d = self.embedding_layer1(self.x)
        self.pretrained_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Assign(self.d),
            trainable=True)
        self.embedding_layer2 = paddle.nn.Embedding(num_embeddings=num_features,
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

        self.linear1 = nn.Linear(num_features, 2, name='seconde_linear')

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

# 创建数据集
class SimpleDataset(paddle.io.Dataset):
    def __init__(self, features, labels=None, is_test=False):
        self.features = features
        self.labels = labels
        self.is_test = is_test
        
    def __getitem__(self, index):
        feature = self.features[index].reshape([-1, self.features.shape[1]])
        if self.labels is not None:
            label = self.labels[index]
            return feature, label
        return feature
        
    def __len__(self):
        return len(self.features)

print("准备数据集...")
# 转换数据为Paddle张量
train_features = paddle.to_tensor(X_train.values.astype('float32'))
train_labels = paddle.to_tensor(y_train.values.astype('int64'))
test_features = paddle.to_tensor(X_test.values.astype('float32'))
test_labels = paddle.to_tensor(y_test.values.astype('int64'))

# 创建数据加载器
batch_size = 32
train_dataset = SimpleDataset(train_features, train_labels)
train_loader = paddle.io.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_dataset = SimpleDataset(test_features, test_labels)
test_loader = paddle.io.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 创建模型实例
print(f"创建AMU模型 (特征数量: {num_features})...")
model = Atten_model(num_features=num_features)

# 定义优化器和损失函数
learning_rate = 0.0001
print(f"使用学习率: {learning_rate}")
optimizer = paddle.optimizer.Adam(
    learning_rate=learning_rate,
    parameters=model.parameters(),
    weight_decay=paddle.regularizer.L2Decay(0.0001)
)
loss_fn = nn.CrossEntropyLoss()

# 训练模型
epochs = 100
print(f"开始训练AMU模型，训练轮数: {epochs}")

# 对训练集做一次标签分布统计
print(f"训练集标签分布: {pd.Series(y_train).value_counts().to_dict()}")

# 定义评估函数
def evaluate_model(model, data_loader, prefix=""):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with paddle.no_grad():
        for data in data_loader():
            if len(data) == 2:
                x, y = data
                logits = model(x)
                probs = F.softmax(logits, axis=1)
                preds = paddle.argmax(probs, axis=1)
                
                all_preds.extend(preds.numpy())
                all_labels.extend(y.numpy())
                all_probs.extend(probs.numpy()[:, 1])  # 保存正类的概率
    
    if all_labels:
        acc = accuracy_score(all_labels, all_preds)
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        # 计算其他指标
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"{prefix} 评估结果:")
        print(f"准确率: {acc:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"混淆矩阵:\n{cm}")
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    return None

# 保存训练过程中的指标
history = {
    'train_loss': [],
    'train_acc': [],
    'val_metrics': []
}

# 开始训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for batch_id, data in enumerate(train_loader()):
        x_data, y_data = data
        
        # 前向传播
        logits = model(x_data)
        loss = loss_fn(logits, y_data)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        # 计算准确率 - 使用argmax直接获取预测类别
        probs = F.softmax(logits, axis=1)
        preds = paddle.argmax(probs, axis=1)
        
        # 收集预测结果用于统计
        all_preds.extend(preds.numpy())
        all_labels.extend(y_data.numpy())
        
        # 统计当前批次的准确率
        batch_correct = (preds == y_data).numpy().sum()
        correct += batch_correct
        total += len(y_data)
        total_loss += float(loss)
    
    # 计算训练集准确率
    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    
    # 保存训练指标
    history['train_loss'].append(avg_loss)
    history['train_acc'].append(train_acc)
    
    # 对验证集进行评估(每10轮或最后一轮)
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        val_metrics = evaluate_model(model, test_loader, prefix=f"Epoch {epoch+1}")
        history['val_metrics'].append(val_metrics)
        
        # 输出训练集预测分布
        unique, counts = np.unique(all_preds, return_counts=True)
        pred_dist = dict(zip(unique, counts))
        print(f"训练集预测分布: {pred_dist}")
        
        # 输出混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        print(f"训练集混淆矩阵:\n{cm}")
    
    # 每轮次输出基本训练信息
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}")

# 最终评估
print("\n开始最终评估AMU模型...")
final_metrics = evaluate_model(model, test_loader, prefix="最终测试集")

# 绘制训练过程中的损失和准确率
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'])
plt.title('训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'])
plt.title('训练准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('amu_training_history.png')
print("训练历史图已保存为 amu_training_history.png")

# 如果有足够的评估数据，绘制ROC曲线
if final_metrics and len(np.unique(final_metrics['labels'])) > 1:
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(final_metrics['labels'], final_metrics['probabilities'])
    roc_auc = roc_auc_score(final_metrics['labels'], final_metrics['probabilities'])
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AMU Model - Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('amu_roc_curve.png')
    print("ROC曲线已保存为 amu_roc_curve.png")
    
    # 计算PR曲线
    precision_curve, recall_curve, _ = precision_recall_curve(
        final_metrics['labels'], final_metrics['probabilities']
    )
    ap = average_precision_score(final_metrics['labels'], final_metrics['probabilities'])
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AP = {ap:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AMU Model - Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('amu_pr_curve.png')
    print("PR曲线已保存为 amu_pr_curve.png")

# 尝试保存模型
try:
    # 创建PaddlePaddle模型对象
    paddle_model = paddle.Model(model)
    # 保存模型
    paddle_model.save('amu_model')
    print("模型已保存为 amu_model")
except Exception as e:
    print(f"保存模型时出错: {e}")

print("\nAMU模型评估完成!")
