"""
AMU模型单独评估脚本
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            precision_recall_curve, roc_curve, average_precision_score)
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.regularizer import L2Decay

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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

# 定义AMU模型
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
        self.encoder1 = nn.TransformerEncoder(self.encoderlayer1, 8)  # [-1,160,10] -->#[-1,10,160]
        self.conv1 = nn.Conv1D(in_channels=20, out_channels=5, kernel_size=1, stride=1, padding=0, data_format='NCL',
                            bias_attr=False)  # [-1,25,160]
        self.bn1 = nn.BatchNorm1D(5)
        self.pool3 = nn.AdaptiveMaxPool1D(1)  # [-1,160,1] -->#[160]

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
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels
        
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
train_dataset = SimpleDataset(train_features, train_labels)
train_loader = paddle.io.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_dataset = SimpleDataset(test_features, test_labels)
test_loader = paddle.io.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# 创建模型实例
print("创建AMU模型...")
model = Atten_model()

# 定义优化器和损失函数
# 使用更高的学习率以加速收敛
optimizer = paddle.optimizer.Adam(
    learning_rate=0.0001,  # 增加学习率
    parameters=model.parameters(),
    weight_decay=paddle.regularizer.L2Decay(0.0001)
)
loss_fn = nn.CrossEntropyLoss()

# 训练模型
epochs = 100  # 增加到100轮训练
print(f"开始训练AMU模型，训练轮数: {epochs}")

# 对训练集做一次标签分布统计
print(f"训练集标签分布: {pd.Series(y_train.values).value_counts().to_dict()}")

# 定义定制的准确率计算函数
def calculate_accuracy(predictions, labels):
    # 将logits转换为概率
    probs = F.softmax(predictions, axis=1)
    # 获取预测类别
    pred_labels = paddle.argmax(probs, axis=1)
    # 计算准确率
    correct = (pred_labels == labels).numpy().sum()
    return correct / len(labels), pred_labels, probs

# 保存每个轮次的准确率统计
epoch_accuracies = []
model.train()

for epoch in range(epochs):
    total_loss = 0
    epoch_correct = 0
    epoch_total = 0
    batch_results = []
    
    # 每10个轮次输出详细的模型预测信息
    detailed_epoch = (epoch % 10 == 0)
    
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
        batch_acc, pred_labels, probs = calculate_accuracy(logits, y_data)
        batch_results.append({
            'batch_id': batch_id,
            'accuracy': batch_acc,
            'loss': float(loss),
            'pred_distribution': np.bincount(pred_labels.numpy(), minlength=2).tolist()
        })
        
        # 累计正确预测数量
        epoch_correct += (pred_labels == y_data).numpy().sum()
        epoch_total += len(y_data)
        total_loss += float(loss)
    
    # 计算并存储该轮次的准确率
    epoch_acc = epoch_correct / epoch_total
    epoch_accuracies.append(epoch_acc)
    
    # 打印每个轮次的结果
    pred_distribution = np.sum([b['pred_distribution'] for b in batch_results], axis=0)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {epoch_acc:.4f}, Pred Dist: {pred_distribution}")
    
    # 每10个轮次输出详细信息
    if detailed_epoch:
        # 在训练集上进行完整的验证
        model.eval()
        all_train_preds = []
        all_train_labels = []
        with paddle.no_grad():
            for data in train_loader():
                x, y = data
                logits = model(x)
                probs = F.softmax(logits, axis=1)
                preds = paddle.argmax(probs, axis=1)
                all_train_preds.extend(preds.numpy())
                all_train_labels.extend(y.numpy())
        
        # 计算回顾性能并显示混淆矩阵
        train_acc = np.mean(np.array(all_train_preds) == np.array(all_train_labels))
        cm = confusion_matrix(all_train_labels, all_train_preds)
        print(f"\n训练集完整评估 - 准确率: {train_acc:.4f}")
        print(f"混淆矩阵:\n{cm}\n")
        model.train()


# 评估模型
print("\n开始评估AMU模型...")
model.eval()
y_probs = []
y_true = []

with paddle.no_grad():
    for batch_id, data in enumerate(test_loader()):
        x_data, y_data = data
        
        # 前向传播
        logits = model(x_data)
        probs = F.softmax(logits, axis=1)
        
        # 收集预测概率和真实标签
        y_probs.append(probs.numpy()[:, 1])  # 取正类的概率
        y_true.append(y_data.numpy())

# 合并所有批次的预测结果
y_prob = np.concatenate(y_probs)
y_true = np.concatenate(y_true)

# 根据阈值获取预测标签
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# 计算评估指标
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 计算ROC曲线
fpr, tpr, _ = roc_curve(y_true, y_prob)

# 计算PR曲线
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

# 打印评估结果
print("\nAMU模型评估结果:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print("Confusion Matrix:")
print(cm)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AMU Model - Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('amu_roc_curve.png')
print("ROC曲线已保存为 amu_roc_curve.png")

# 绘制PR曲线
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
