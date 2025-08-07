"""
AMU Model Evaluation Script - Using Original Data Loading Method
Implemented exactly according to the original code provided by the user
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

# ------------ Implement data loading exactly according to user-provided code ------------
print("Starting data loading...")
try:
    # Original path was 'data/data156006/logfourupsample.csv', but we've changed it to local path
    data = pd.read_csv('logfourupsample.csv', sep=',')
    print(f"Successfully loaded data, shape: {data.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Organize dataset, split test and training sets, using original random seed 1000
x, y = data.iloc[:, :-1], data.iloc[:, -1]
train_x, test_x, train_y, test_y = ms_train_test_split(x, y, test_size=0.2, random_state=1000)
print(f"Training set shape: {train_x.shape}, Test set shape: {test_x.shape}")
print(f"Label distribution - Training set: {train_y.value_counts().to_dict()}, Test set: {test_y.value_counts().to_dict()}")

# Prepare test dataset, exactly according to original code
testdata = pd.concat([test_x, test_y], axis=1)
data_np = np.array(testdata).astype('float32')
selfdata = []
for i in range(data_np.shape[0]):
    input_np = data_np[i, :-1].reshape([-1, 160])
    label_np = data_np[i, -1].astype('int64')
    selfdata.append([input_np, label_np])
testdata = selfdata

# Custom DL dataset - completely consistent with original code
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

# Create test dataset
test_data = MyDataset(mode='test')
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Prepare similar processing for training set
train_data = pd.concat([train_x, train_y], axis=1)
train_np = np.array(train_data).astype('float32')
train_selfdata = []
for i in range(train_np.shape[0]):
    input_np = train_np[i, :-1].reshape([-1, 160])
    label_np = train_np[i, -1].astype('int64')
    train_selfdata.append([input_np, label_np])

# Extend MyDataset class to support training mode
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

# Create training dataset
train_dataset = ExtendedDataset(train_data=train_selfdata, mode='train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ------------ Implement AMU model exactly according to user-provided code ------------
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

# Create model instance
print("Creating AMU model...")
model = Atten_model()

# Training parameters - use higher learning rate to accelerate convergence
learning_rate = 0.0001
print(f"Using learning rate: {learning_rate}")
optimizer = paddle.optimizer.Adam(
    learning_rate=learning_rate,
    parameters=model.parameters(),
    weight_decay=paddle.regularizer.L2Decay(0.0001)
)
loss_fn = nn.CrossEntropyLoss()

# Verify model architecture
print("Model architecture:")
model_params = 0
for name, param in model.named_parameters():
    print(f"  - {name}: {param.shape}")
    model_params += np.prod(param.shape)
print(f"Total model parameters: {model_params:,}")

# Training and evaluation function
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
    
    # Calculate metrics
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

# Train model
epochs = 100
print(f"Starting AMU model training, number of epochs: {epochs}")

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch_id, data in enumerate(train_loader):
        x_data, y_data = data
        
        # Forward pass
        logits = model(x_data)
        loss = loss_fn(logits, y_data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        total_loss += float(loss)
        batch_count += 1
    
    # Evaluate after each epoch
    avg_loss = total_loss / batch_count
    
    # Evaluate test set every 10 epochs or last epoch
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        # Evaluate training set
        train_metrics = evaluate_model(model, train_loader)
        # Evaluate test set
        test_metrics = evaluate_model(model, test_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        print(f"  Training set - Accuracy: {train_metrics['accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Test set - Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
        print(f"  Training set confusion matrix:\n{train_metrics['confusion_matrix']}")
        print(f"  Test set confusion matrix:\n{test_metrics['confusion_matrix']}")
    else:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Final evaluation
final_train_metrics = evaluate_model(model, train_loader)
final_test_metrics = evaluate_model(model, test_loader)

print("\nTraining completed!")
print(f"Final training set metrics:")
print(f"  Accuracy: {final_train_metrics['accuracy']:.4f}")
print(f"  Precision: {final_train_metrics['precision']:.4f}")
print(f"  Recall: {final_train_metrics['recall']:.4f}")
print(f"  F1 score: {final_train_metrics['f1']:.4f}")
print(f"  Confusion matrix:\n{final_train_metrics['confusion_matrix']}")

print(f"\nFinal test set metrics:")
print(f"  Accuracy: {final_test_metrics['accuracy']:.4f}")
print(f"  Precision: {final_test_metrics['precision']:.4f}")
print(f"  Recall: {final_test_metrics['recall']:.4f}")
print(f"  F1 score: {final_test_metrics['f1']:.4f}")
print(f"  Confusion matrix:\n{final_test_metrics['confusion_matrix']}")

# Save model
try:
    paddle_model = paddle.Model(model)
    paddle_model.save('amu_original_model')
    print("Model saved as amu_original_model")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nAMU model evaluation completed!")
