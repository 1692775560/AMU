"""
AMU Model Exact Implementation and Evaluation
Using exactly the same AMU model code as provided by the user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.regularizer import L2Decay

print("Starting to load data...")
# Load data
try:
    data = pd.read_csv('logfourupsample.csv')
    print(f"Successfully loaded data, shape: {data.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Separate features and labels
if 'target' in data.columns:
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
else:
    # Assume the last column is the label
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

print(f"Feature shape: {X.shape}, Label shape: {y.shape}")
print(f"Label distribution: {y.value_counts().to_dict()}")

# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# ------------ Implement AMU model exactly as provided by user ------------
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

# Create dataset
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

print("Preparing dataset...")
# Convert data to Paddle tensors
train_features = paddle.to_tensor(X_train.values.astype('float32'))
train_labels = paddle.to_tensor(y_train.values.astype('int64'))
test_features = paddle.to_tensor(X_test.values.astype('float32'))
test_labels = paddle.to_tensor(y_test.values.astype('int64'))

# Create data loaders
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

# Validate model architecture
print("Model architecture:")
model_params = 0
for name, param in model.named_parameters():
    print(f"  - {name}: {param.shape}")
    model_params += np.prod(param.shape)
print(f"Total model parameters: {model_params:,}")

# Train model
epochs = 100
print(f"Starting AMU model training, epochs: {epochs}")

# Training and evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with paddle.no_grad():
        for data in data_loader():
            x, y = data
            logits = model(x)
            probs = F.softmax(logits, axis=1)
            preds = paddle.argmax(probs, axis=1)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, cm

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch_id, data in enumerate(train_loader()):
        x_data, y_data = data
        
        # Forward propagation
        logits = model(x_data)
        loss = loss_fn(logits, y_data)
        
        # Backward propagation
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        total_loss += float(loss)
        batch_count += 1
    
    # Evaluate after each epoch
    avg_loss = total_loss / batch_count
    
    # Evaluate test set every 5 epochs or last epoch
    if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
        train_acc, train_cm = evaluate_model(model, train_loader)
        test_acc, test_cm = evaluate_model(model, test_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        print(f"Train CM:\n{train_cm}\nTest CM:\n{test_cm}\n")
    else:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Final evaluation
train_acc, train_cm = evaluate_model(model, train_loader)
test_acc, test_cm = evaluate_model(model, test_loader)

print("\nTraining completed!")
print(f"Final training accuracy: {train_acc:.4f}")
print(f"Final test accuracy: {test_acc:.4f}")
print(f"Training confusion matrix:\n{train_cm}")
print(f"Test confusion matrix:\n{test_cm}")

# Save model
try:
    paddle_model = paddle.Model(model)
    paddle_model.save('amu_exact_model')
    print("Model saved as amu_exact_model")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nAMU model evaluation completed!")
