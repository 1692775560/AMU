"""
Multi-model evaluation script - Evaluate multiple models using the same data loading approach
Includes AMU model and CNN model
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

# Create results directory
os.makedirs('model_results', exist_ok=True)

# ------------ Data loading implementation as provided by user ------------
print("Starting data loading...")
try:
    data = pd.read_csv('logfourupsample.csv', sep=',')
    print(f"Successfully loaded data, shape: {data.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Organize dataset, split into test and training sets, using original random seed 1000
x, y = data.iloc[:, :-1], data.iloc[:, -1]
train_x, test_x, train_y, test_y = ms_train_test_split(x, y, test_size=0.2, random_state=1000)
print(f"Training set shape: {train_x.shape}, Test set shape: {test_x.shape}")
print(f"Label distribution - Training set: {train_y.value_counts().to_dict()}, Test set: {test_y.value_counts().to_dict()}")

# Prepare test dataset
testdata = pd.concat([test_x, test_y], axis=1)
data_np = np.array(testdata).astype('float32')
selfdata = []
for i in range(data_np.shape[0]):
    input_np = data_np[i, :-1].reshape([-1, 160])
    label_np = data_np[i, -1].astype('int64')
    selfdata.append([input_np, label_np])
testdata = selfdata

# Prepare similar processing for training set
train_data = pd.concat([train_x, train_y], axis=1)
train_np = np.array(train_data).astype('float32')
train_selfdata = []
for i in range(train_np.shape[0]):
    input_np = train_np[i, :-1].reshape([-1, 160])
    label_np = train_np[i, -1].astype('int64')
    train_selfdata.append([input_np, label_np])

# Custom dataset class
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

# Create training and test datasets
train_dataset = CustomDataset(train_selfdata, mode='train')
test_dataset = CustomDataset(selfdata, mode='test')

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------ Model definitions ------------
# 1. AMU model definition
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

# 2. CNN model definition
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
        # CNN model expects input shape [batch_size, channels, seq_len]
        # Current input is [batch_size, seq_len], need to add a channel dimension
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

# 3. MLP model definition
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

# ------------ Training and evaluation functions ------------
def train_model(model, model_name, train_loader, test_loader, epochs=100, learning_rate=0.0001, weight_decay=0.0001):
    """
    Train model and record training process
    """
    print(f"\nStarting training {model_name} model...")
    
    # Initialize optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=learning_rate,
        parameters=model.parameters(),
        weight_decay=L2Decay(weight_decay)
    )
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Record training history
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
    
    # Training loop
    start_time = time.time()
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
        
        # Calculate average loss
        avg_loss = total_loss / batch_count
        history['train_loss'].append(avg_loss)
        
        # Evaluate every 10 epochs or on the last epoch
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            # Evaluate training set
            train_metrics = evaluate_model(model, train_loader)
            # Evaluate test set
            test_metrics = evaluate_model(model, test_loader)
            
            # Record metrics
            history['train_acc'].append(train_metrics['accuracy'])
            history['test_acc'].append(test_metrics['accuracy'])
            history['train_precision'].append(train_metrics['precision'])
            history['train_recall'].append(train_metrics['recall'])
            history['train_f1'].append(train_metrics['f1'])
            history['test_precision'].append(test_metrics['precision'])
            history['test_recall'].append(test_metrics['recall'])
            history['test_f1'].append(test_metrics['f1'])
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            print(f"  Training set - Accuracy: {train_metrics['accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Test set - Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
            print(f"  Training set confusion matrix:\n{train_metrics['confusion_matrix']}")
            print(f"  Test set confusion matrix:\n{test_metrics['confusion_matrix']}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Training time
    training_time = time.time() - start_time
    print(f"{model_name} model training completed! Training time: {training_time:.2f} seconds")
    
    # Final evaluation
    final_train_metrics = evaluate_model(model, train_loader)
    final_test_metrics = evaluate_model(model, test_loader)
    
    # Calculate ROC and PR curves
    test_labels, test_preds, test_probs = get_predictions(model, test_loader)
    
    # Calculate ROC curve data
    fpr, tpr, _ = roc_curve(test_labels, test_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Calculate PR curve data
    precision, recall, _ = precision_recall_curve(test_labels, test_probs[:, 1])
    pr_auc = auc(recall, precision)
    
    # Plot and save ROC curve
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
    
    # Plot and save PR curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(f'model_results/{model_name}_pr_curve.png')
    
    # Plot training history
    plt.figure(figsize=(12, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
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
    Evaluate model performance
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
    Get model predictions, true labels, and prediction probabilities
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

# ------------ Main function: Train and evaluate multiple models ------------
def main():
    print("Starting evaluation of multiple models...")
    
    # 存储所有模型的结果
    all_results = {}
    
    # Define model list
    models = {
        'AMU': Atten_model(),
        'CNN': CNN_model(),
        'MLP': MLP_model()
    }
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training and evaluating {model_name} model")
        print(f"{'='*50}")
        
        # Train model
        metrics, history = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=100,
            learning_rate=0.0001
        )
        
        # Store results
        all_results[model_name] = {
            'metrics': metrics,
            'history': history
        }
        
        # Output model performance
        print(f"\n{model_name} model final performance:")
        print(f"Training set accuracy: {metrics['train']['accuracy']:.4f}")
        print(f"Test set accuracy: {metrics['test']['accuracy']:.4f}")
        print(f"Test set precision: {metrics['test']['precision']:.4f}")
        print(f"Test set recall: {metrics['test']['recall']:.4f}")
        print(f"Test set F1 score: {metrics['test']['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")
        print(f"Training time: {metrics['training_time']:.2f} seconds")
    
    # Compare performance of all models
    print("\nAll models performance comparison:")
    print("="*80)
    print(f"{'Model Name':<12}{'Train Acc':<12}{'Test Acc':<12}{'Precision':<12}{'Recall':<10}{'F1 Score':<10}{'ROC AUC':<10}{'PR AUC':<10}{'Train Time(s)':<15}")
    print("-"*80)
    
    for model_name, result in all_results.items():
        metrics = result['metrics']
        print(f"{model_name:<12}{metrics['train']['accuracy']:.4f}{'    '}{metrics['test']['accuracy']:.4f}{'    '}{metrics['test']['precision']:.4f}{'      '}{metrics['test']['recall']:.4f}{'    '}{metrics['test']['f1']:.4f}{'    '}{metrics['roc_auc']:.4f}{'    '}{metrics['pr_auc']:.4f}{'    '}{metrics['training_time']:.2f}")
    
    print("="*80)
    
    # Plot ROC curve comparison for all models
    plt.figure(figsize=(12, 10))
    
    for model_name, model in models.items():
        # Get test set predictions
        test_labels, test_preds, test_probs = get_predictions(model, test_loader)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(test_labels, test_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.savefig('model_results/all_models_roc_comparison.png')
    
    # Plot accuracy comparison for all models
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
    
    # Add numerical labels on bar chart
    for i, v in enumerate(train_accs):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(test_accs):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.savefig('model_results/all_models_accuracy_comparison.png')
    
    print("\nAll charts and models have been saved to model_results directory")
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
