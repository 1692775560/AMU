"""
AMU模型评估脚本 - 使用原始数据加载方式
完全按照用户提供的原始代码实现
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as ms_train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.regularizer import L2Decay
from paddle.io import Dataset, DataLoader

# ------------ 完全按照用户提供的代码实现数据加载 ------------
print("开始加载数据...")
try:
    # 原始路径是'data/data156006/logfourupsample.csv'，但我们已修改为本地路径
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

# 准备测试数据集，完全按照原始代码
testdata = pd.concat([test_x, test_y], axis=1)
data_np = np.array(testdata).astype('float32')
selfdata = []
for i in range(data_np.shape[0]):
    input_np = data_np[i, :-1].reshape([-1, 160])
    label_np = data_np[i, -1].astype('int64')
    selfdata.append([input_np, label_np])
testdata = selfdata

# 自定义DL数据集 - 与原始代码完全一致
class MyDataset(Dataset):
    def __init__(self, mode='train'):
        super(MyDataset, self).__init__()
        if mode == "test":
            self.data = testdata
            
    def __getitem__(self, index):
        data = self.data[index][0]
        label = self.data[index][1]
        return data, label
    
    def __len__(self):
        return len(self.data)

# 创建测试数据集
test_data = MyDataset(mode='test')
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 为训练集准备类似的处理
train_data = pd.concat([train_x, train_y], axis=1)
train_np = np.array(train_data).astype('float32')
train_selfdata = []
for i in range(train_np.shape[0]):
    input_np = train_np[i, :-1].reshape([-1, 160])
    label_np = train_np[i, -1].astype('int64')
    train_selfdata.append([input_np, label_np])

# 扩展MyDataset类以支持训练模式
class ExtendedDataset(Dataset):
    def __init__(self, train_data=None, test_data=None, mode='train'):
        super(ExtendedDataset, self).__init__()
        self.mode = mode
        if mode == "train" and train_data is not None:
            self.data = train_data
        elif mode == "test" and test_data is not None:
            self.data = test_data
    
    def __getitem__(self, index):
        data = self.data[index][0]
        label = self.data[index][1]
        return data, label
    
    def __len__(self):
        return len(self.data)

# 创建训练数据集
train_dataset = ExtendedDataset(train_data=train_selfdata, mode='train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ------------ 完全按照用户提供的代码实现AMU模型 ------------
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

# 创建模型实例
print("创建AMU模型...")
model = Atten_model()

# 训练参数 - 使用较高的学习率以加速收敛
learning_rate = 0.0001
print(f"使用学习率: {learning_rate}")
optimizer = paddle.optimizer.Adam(
    learning_rate=learning_rate,
    parameters=model.parameters(),
    weight_decay=paddle.regularizer.L2Decay(0.0001)
)
loss_fn = nn.CrossEntropyLoss()

# 验证模型架构
print("模型架构:")
model_params = 0
for name, param in model.named_parameters():
    print(f"  - {name}: {param.shape}")
    model_params += np.prod(param.shape)
print(f"模型总参数数量: {model_params:,}")

# 训练与评估函数
def evaluate_model(model, data_loader):
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

# 训练模型
epochs = 100
print(f"开始训练AMU模型，训练轮数: {epochs}")

# 训练循环
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
    
    # 每个epoch结束后评估
    avg_loss = total_loss / batch_count
    
    # 每10个epoch或最后一个epoch评估一次测试集
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        # 评估训练集
        train_metrics = evaluate_model(model, train_loader)
        # 评估测试集
        test_metrics = evaluate_model(model, test_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        print(f"  训练集 - 准确率: {train_metrics['accuracy']:.4f}, 精确率: {train_metrics['precision']:.4f}, 召回率: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  测试集 - 准确率: {test_metrics['accuracy']:.4f}, 精确率: {test_metrics['precision']:.4f}, 召回率: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
        print(f"  训练集混淆矩阵:\n{train_metrics['confusion_matrix']}")
        print(f"  测试集混淆矩阵:\n{test_metrics['confusion_matrix']}")
    else:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 最终评估
final_train_metrics = evaluate_model(model, train_loader)
final_test_metrics = evaluate_model(model, test_loader)

print("\n训练完成!")
print(f"最终训练集指标:")
print(f"  准确率: {final_train_metrics['accuracy']:.4f}")
print(f"  精确率: {final_train_metrics['precision']:.4f}")
print(f"  召回率: {final_train_metrics['recall']:.4f}")
print(f"  F1分数: {final_train_metrics['f1']:.4f}")
print(f"  混淆矩阵:\n{final_train_metrics['confusion_matrix']}")

print(f"\n最终测试集指标:")
print(f"  准确率: {final_test_metrics['accuracy']:.4f}")
print(f"  精确率: {final_test_metrics['precision']:.4f}")
print(f"  召回率: {final_test_metrics['recall']:.4f}")
print(f"  F1分数: {final_test_metrics['f1']:.4f}")
print(f"  混淆矩阵:\n{final_test_metrics['confusion_matrix']}")

# 保存模型
try:
    paddle_model = paddle.Model(model)
    paddle_model.save('amu_original_model')
    print("模型已保存为 amu_original_model")
except Exception as e:
    print(f"保存模型时出错: {e}")

print("\nAMU模型评估完成!")
