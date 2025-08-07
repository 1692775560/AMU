"""
AMU Model Standalone Evaluation Script - Fixed Version
Using the exact AMU model structure provided by the user
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

# Set random seed to ensure reproducible results
np.random.seed(42)
paddle.seed(42)

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

# Ensure feature count is 160, otherwise adjust model
num_features = X.shape[1]
print(f"Number of features: {num_features}")

# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Define AMU model - using exact code provided by user
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

# Create dataset
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

print("Preparing datasets...")
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
print(f"Creating AMU model (number of features: {num_features})...")
model = Atten_model(num_features=num_features)

# Define optimizer and loss function
learning_rate = 0.0001
print(f"Using learning rate: {learning_rate}")
optimizer = paddle.optimizer.Adam(
    learning_rate=learning_rate,
    parameters=model.parameters(),
    weight_decay=paddle.regularizer.L2Decay(0.0001)
)
loss_fn = nn.CrossEntropyLoss()

# Train model
epochs = 100
print(f"Starting AMU model training, number of epochs: {epochs}")

# Perform label distribution statistics on training set
print(f"Training set label distribution: {pd.Series(y_train).value_counts().to_dict()}")

# Define evaluation function
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
                all_probs.extend(probs.numpy()[:, 1])  # Save positive class probabilities
    
    if all_labels:
        acc = accuracy_score(all_labels, all_preds)
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        # Calculate other metrics
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"{prefix} Evaluation results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(f"Confusion matrix:\n{cm}")
        
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

# Save metrics during training process
history = {
    'train_loss': [],
    'train_acc': [],
    'val_metrics': []
}

# Start training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for batch_id, data in enumerate(train_loader()):
        x_data, y_data = data
        
        # Forward pass
        logits = model(x_data)
        loss = loss_fn(logits, y_data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        # Calculate accuracy - use argmax to directly get predicted classes
        probs = F.softmax(logits, axis=1)
        preds = paddle.argmax(probs, axis=1)
        
        # Collect prediction results for statistics
        all_preds.extend(preds.numpy())
        all_labels.extend(y_data.numpy())
        
        # Calculate accuracy for current batch
        batch_correct = (preds == y_data).numpy().sum()
        correct += batch_correct
        total += len(y_data)
        total_loss += float(loss)
    
    # Calculate training set accuracy
    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    
    # Save training metrics
    history['train_loss'].append(avg_loss)
    history['train_acc'].append(train_acc)
    
    # Evaluate on validation set (every 10 epochs or last epoch)
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        val_metrics = evaluate_model(model, test_loader, prefix=f"Epoch {epoch+1}")
        history['val_metrics'].append(val_metrics)
        
        # Output training set prediction distribution
        unique, counts = np.unique(all_preds, return_counts=True)
        pred_dist = dict(zip(unique, counts))
        print(f"Training set prediction distribution: {pred_dist}")
        
        # Output confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(f"Training set confusion matrix:\n{cm}")
    
    # Output basic training information for each epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}")

# Final evaluation
print("\nStarting final evaluation of AMU model...")
final_metrics = evaluate_model(model, test_loader, prefix="Final test set")

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'])
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('amu_training_history.png')
print("Training history plot saved as amu_training_history.png")

# If there is enough evaluation data, plot ROC curve
if final_metrics and len(np.unique(final_metrics['labels'])) > 1:
    # Calculate ROC curve
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
    print("ROC curve saved as amu_roc_curve.png")
    
    # Calculate PR curve
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
