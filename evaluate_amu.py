"""
AMU Model Standalone Evaluation Script
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

print("Starting data loading...")
# 加载数据
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
    # Assume last column is the label
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
print(f"Label distribution: {y.value_counts().to_dict()}")

# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Define AMU model
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

print("Preparing datasets...")
# Convert data to Paddle tensors
train_features = paddle.to_tensor(X_train.values.astype('float32'))
train_labels = paddle.to_tensor(y_train.values.astype('int64'))
test_features = paddle.to_tensor(X_test.values.astype('float32'))
test_labels = paddle.to_tensor(y_test.values.astype('int64'))

# Create data loaders
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

# Create model instance
print("Create AMU model instance...")
model = Atten_model()

# Initialize optimizer and loss function
# Use a higher learning rate to accelerate convergence
optimizer = paddle.optimizer.Adam(
    learning_rate=0.0001,  # Increase learning rate
    parameters=model.parameters(),
    weight_decay=paddle.regularizer.L2Decay(0.0001)
)
loss_fn = nn.CrossEntropyLoss()

# Set training parameters
epochs = 100  # Increase to 100 training epochs
print(f"Start training AMU model, number of epochs: {epochs}")

# Perform label distribution statistics on the training set
print(f"Training set label distribution: {pd.Series(y_train.values).value_counts().to_dict()}")

# Define a custom accuracy calculation function
# Define custom accuracy calculation function
def calculate_accuracy(predictions, labels):
    # Convert logits to probabilities
    probs = F.softmax(predictions, axis=1)
    # Get predicted labels
    pred_labels = paddle.argmax(probs, axis=1)
    # Calculate accuracy
    correct = (pred_labels == labels).numpy().sum()
    return correct / len(labels), pred_labels, probs

# Save accuracy statistics for each epoch
epoch_accuracies = []
model.train()

for epoch in range(epochs):
    total_loss = 0
    epoch_correct = 0
    epoch_total = 0
    batch_results = []
    
    # Output detailed model prediction information every 10 epochs
    detailed_epoch = (epoch % 10 == 0)
    
    for batch_id, data in enumerate(train_loader()):
        x_data, y_data = data
        
        # Forward pass
        logits = model(x_data)
        loss = loss_fn(logits, y_data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        # Calculate accuracy
        batch_acc, pred_labels, probs = calculate_accuracy(logits, y_data)
        batch_results.append({
            'batch_id': batch_id,
            'accuracy': batch_acc,
            'loss': float(loss),
            'pred_distribution': np.bincount(pred_labels.numpy(), minlength=2).tolist()
        })
        
        # Accumulate correct predictions
        epoch_correct += (pred_labels == y_data).numpy().sum()
        epoch_total += len(y_data)
        total_loss += float(loss)
    
    # Calculate and store accuracy for this epoch
    epoch_acc = epoch_correct / epoch_total
    epoch_accuracies.append(epoch_acc)
    
    # Print results for each epoch
    pred_distribution = np.sum([b['pred_distribution'] for b in batch_results], axis=0)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {epoch_acc:.4f}, Pred Dist: {pred_distribution}")
    
    # Output detailed information every 10 epochs
    if detailed_epoch:
        # Perform complete validation on training set
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
        
        # Calculate retrospective performance and display confusion matrix
        train_acc = np.mean(np.array(all_train_preds) == np.array(all_train_labels))
        cm = confusion_matrix(all_train_labels, all_train_preds)
        print(f"\nComplete training set evaluation - Accuracy: {train_acc:.4f}")
        print(f"Confusion matrix:\n{cm}\n")
        model.train()


# Evaluate model
print("\nStarting AMU model evaluation...")
model.eval()
y_probs = []
y_true = []

with paddle.no_grad():
    for batch_id, data in enumerate(test_loader()):
        x_data, y_data = data
        
        # Forward pass
        logits = model(x_data)
        probs = F.softmax(logits, axis=1)
        
        # Collect prediction probabilities and true labels
        y_probs.append(probs.numpy()[:, 1])  # Take positive class probability
        y_true.append(y_data.numpy())

# Merge prediction results from all batches
y_prob = np.concatenate(y_probs)
y_true = np.concatenate(y_true)

# Get predicted labels based on threshold
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# Calculate evaluation metrics
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_true, y_prob)

# Calculate PR curve
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

# Print evaluation results
print("\nAMU Model Evaluation Results:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print("Confusion Matrix:")
print(cm)

# Plot ROC curve
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
print("ROC curve saved as amu_roc_curve.png")

# Plot PR curve
plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AP = {ap:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AMU Model - Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('amu_pr_curve.png')
print("PR curve saved as amu_pr_curve.png")

# Try to save model
try:
    # Create PaddlePaddle model object
    paddle_model = paddle.Model(model)
    # Save model
    paddle_model.save('amu_model')
    print("Model saved as amu_model")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nAMU model evaluation completed!")
