import pandas as pd
import numpy as np
import sklearn.metrics as sm
# import scikitplot as skplt
import sklearn
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.ensemble as se
import sklearn.tree as st
import xgboost
import paddle
import paddle.nn as nn
import paddle.metric as metric
import pandas as pd
from paddle.io import Dataset,DataLoader
import numpy as np
import paddle.nn.functional as F
from paddle.regularizer import L2Decay
import matplotlib.pylab as pl
from matplotlib import cm
import sklearn.ensemble as se
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize    
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score,roc_curve
train_params={'lr_rate':0.00001,
              'epoch_num':2000,
              'L2Decay_rate':0.0,
              'random seed':12
              }


### ----------- ### 
data=pd.read_csv('logfourupsample.csv',sep=',')
# 整理数据集，拆分测试集训练集
x, y = data.iloc[:, :-1], data.iloc[:,-1]
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.2, random_state=1000)
testdata=pd.concat([test_x,test_y],axis=1)
data_np=np.array(testdata).astype('float32')
selfdata=[]
for i in range(data_np.shape[0]):
    input_np=data_np[i, :-1].reshape([-1,160])
    label_np=data_np[i, -1].astype('int64')
    selfdata.append([input_np,label_np])
testdata=selfdata
# 自定义DL数据集
class MyDataset(Dataset):
    def __init__(self,mode='train'):
        super(MyDataset, self).__init__()

        if mode == "test":
            self.data =testdata
    def __getitem__(self, index):
            data = self.data[index][0]
            label = self.data[index][1]
            return data, label
    def __len__(self):
        return  len(self.data)
test_data=MyDataset(mode='test')



#--------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# 读取数据（假设有两个数据集）
data1=pd.read_csv('logfourupsample.csv',sep=',')

data2 = pd.read_csv('four.csv')  # 第二个数据集（这里用同一个文件示例）
from matplotlib.colors import ListedColormap
custom_cmap1 = ListedColormap(sns.color_palette("coolwarm", 100))
custom_cmap2 = ListedColormap(sns.color_palette("viridis", 100))
# 设置绘图风格
sns.set(style="whitegrid", font_scale=1.2)  # 使用白色网格背景，并调整字体大小
plt.style.use('seaborn')

# 创建画布和子图
fig, axes = plt.subplots(1, 2, figsize=(30, 15))  # 1行2列，并排显示

# 绘制第一张热力图
sns.heatmap(
    data1,
    cmap="mako",  # 使用高级颜色映射
    annot=False,
    fmt=".2f",
    linewidths=0.5,
    linecolor='lightgray',  # 网格线颜色
    cbar_kws={"shrink": 0.8, "label": "Expression Level"},
    square=True,
    ax=axes[0]  # 指定绘制在第一个子图
)
axes[0].set_title('Heatmap 1: Gene Expression', fontsize=20, pad=20, fontweight='bold')
axes[0].set_xlabel('Genes', fontsize=15, labelpad=15, fontweight='bold')
axes[0].set_ylabel('Samples', fontsize=15, labelpad=15, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=10, horizontalalignment='center')
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=10)

# 隐藏部分标签（每隔5个显示一个）
for i, label in enumerate(axes[0].get_xticklabels()):
    if i % 5 != 0:
        label.set_visible(False)
for i, label in enumerate(axes[0].get_yticklabels()):
    if i % 5 != 0:
        label.set_visible(False)

# 调整颜色条
cbar1 = axes[0].collections[0].colorbar
cbar1.ax.tick_params(labelsize=12)
cbar1.set_label('Expression Level', fontsize=14, rotation=270, labelpad=20, fontweight='bold')

# 添加注释（可选）
axes[0].annotate(
    'Important Region',  # 注释文本
    xy=(10, 5),          # 箭头指向的位置
    xytext=(15, 10),     # 文本位置
    arrowprops=dict(facecolor='red', shrink=0.05),  # 箭头样式
    fontsize=12,
    color='red'
)

# 绘制第二张热力图
sns.heatmap(
    data2,
    cmap="rocket",  # 使用不同的高级颜色映射
    annot=False,
    fmt=".2f",
    linewidths=0.5,
    linecolor='lightgray',  # 网格线颜色
    cbar_kws={"shrink": 0.8, "label": "Expression Level"},
    square=True,
    ax=axes[1]  # 指定绘制在第二个子图
)
axes[1].set_title('Heatmap 2: Gene Expression', fontsize=20, pad=20, fontweight='bold')
axes[1].set_xlabel('Genes', fontsize=15, labelpad=15, fontweight='bold')
axes[1].set_ylabel('Samples', fontsize=15, labelpad=15, fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, fontsize=10, horizontalalignment='center')
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=10)

# 隐藏部分标签（每隔5个显示一个）
for i, label in enumerate(axes[1].get_xticklabels()):
    if i % 5 != 0:
        label.set_visible(False)
for i, label in enumerate(axes[1].get_yticklabels()):
    if i % 5 != 0:
        label.set_visible(False)

# 调整颜色条
cbar2 = axes[1].collections[0].colorbar
cbar2.ax.tick_params(labelsize=12)
cbar2.set_label('Expression Level', fontsize=14, rotation=270, labelpad=20, fontweight='bold')

# 添加注释（可选）
axes[1].annotate(
    'Key Cluster',  # 注释文本
    xy=(20, 15),    # 箭头指向的位置
    xytext=(25, 20),# 文本位置
    arrowprops=dict(facecolor='blue', shrink=0.05),  # 箭头样式
    fontsize=12,
    color='blue'
)

# 调整布局
plt.tight_layout()

# 保存图形（可选）
plt.savefig('combined_heatmaps_beautified.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()



#-----------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据（假设有两个数据集）
# 读取数据
data1 = pd.read_csv('logfourupsample.csv',sep=',')
data2 = pd.read_csv('four.csv')  # 第二个数据集（这里用同一个文件示例）



#------#
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
#建模
atten_model = Atten_model()
attenmodel=paddle.Model(atten_model)
opt=paddle.optimizer.Adam(learning_rate=train_params['lr_rate'],parameters=attenmodel.parameters(),
                weight_decay=L2Decay(0.0001))
# 不加载预训练参数，直接训练模型
print("使用随机初始化参数开始训练AMU模型...")

# 添加保存模型的代码
# 在训练完成后可以使用以下代码保存模型
# attenmodel.save('test02-04')  # 取消注释此行来保存模型


#------#
class CnnModel(nn.Layer):
    def __init__(self):
        super(CnnModel, self).__init__()#[-1,1,160]
        self.drop=nn.Dropout(0.5)
        self.relu=nn.ReLU()        
        self.softmax=nn.Softmax()
        self.flatten=nn.Flatten(1,-1)
        self.bn1=nn.BatchNorm1D(10)
        self.bn2=nn.BatchNorm1D(5)
        self.conv1 =nn.Conv1D(in_channels=1, out_channels=10, kernel_size=1,
                                              stride=1, padding=0, data_format='NCL', bias_attr=True)
        self.conv2 =nn.Conv1D(in_channels=10, out_channels=10, kernel_size=2,
                                              stride=2, padding=0, data_format='NCL', bias_attr=True)     
        self.conv3 =nn.Conv1D(in_channels=10, out_channels=5, kernel_size=2,
                                              stride=2, padding=0, data_format='NCL', bias_attr=True)                                                                           
        self.linear1=nn.Linear(200,2,name='seconde_linear')
        self.pool2=nn.AdaptiveMaxPool1D(1)
        self.pool1=nn.AdaptiveMaxPool1D(5)
    def forward(self, x):  
        x=self.conv1(x)   
            #[-1,10,160]
        x=self.relu(x) 
        x =self.bn1(x)    
        x=self.conv2(x)#[-1,10,80]
        x =self.bn1(x)        
        x=self.relu(x)  
        x=self.conv3(x)#[-1,5,40]
        x =self.bn2(x)        
        x=self.relu(x)  
        x= x.transpose((0, 2, 1))    #[-1,40,5]
        x=self.flatten(x)        
        x=self.linear1(x) 
        x=self.softmax(x)
        return x
cnnmodel=CnnModel()
cnnmodel=paddle.Model(cnnmodel)
# 不加载预训练参数，直接训练模型
print("使用随机初始化参数开始训练CNN模型...")

# 添加保存模型的代码
# 在训练完成后可以使用以下代码保存模型
# cnnmodel.save('1851')  # 取消注释此行来保存模型



#------#
svmmodel = svm.SVC(C=1, gamma=1,kernel='rbf',probability=True)
treemodel = st.DecisionTreeClassifier(max_depth=4)
model_adaboost = se.AdaBoostClassifier(treemodel, random_state=7)
models = [ ('SVM',svmmodel), 
           ('RandomForest',se.RandomForestClassifier(random_state=12,max_depth=4,n_estimators=80) ),
           ('AdaBoost', model_adaboost),
           ('XGBoost', xgboost.XGBClassifier()) ]
cnn_models= [ ('AMU',attenmodel),
            ('CNN',cnnmodel)]

#--------#
def calculate_auc(y_test, pred):
    print("auc:",roc_auc_score(y_test, pred))
    fpr, tpr, thersholds = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k-', label='ROC (area = {0:.2f})'.format(roc_auc),color='blue', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()
    
#使用Yooden法寻找最佳阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
 
#计算roc值
def ROC(label, y_prob):
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    roc_auc = auc(fpr, tpr)
    optimal_threshold, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_threshold, optimal_point
 
#计算混淆矩阵
def calculate_metric(label, y_prob,optimal_threshold):
    p=[]
    for i in y_prob:
        if i>=optimal_threshold:
            p.append(1)
        else:
            p.append(0)
    confusion = confusion_matrix(label,p)
    print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    Accuracy=(TP+TN)/float(TP+TN+FP+FN)
    Sensitivity=TP / float(TP+FN)
    Specificity=TN / float(TN+FP)
    return Accuracy,Sensitivity,Specificity

#---------#
#循环训练模型
results=[]
roc_=[]
for name,model in models:
    clf=model.fit(train_x,train_y)
    pred_proba = clf.predict_proba(test_x)
    y_prob=pred_proba[:,1]
    fpr, tpr, roc_auc, optimal_threshold, optimal_point=ROC(test_y, y_prob)
    Accuracy,Sensitivity,Specificity=calculate_metric(test_y, y_prob,optimal_threshold)
    result=[optimal_threshold,Accuracy,Sensitivity,Specificity,roc_auc,name]
    results.append(result)
    roc_.append([fpr,tpr,roc_auc,name])

for name,model in cnn_models:   
    x=[test_data.__getitem__(i)[0] for i in range(test_data.__len__())]
    lbl=[test_data.__getitem__(i)[1] for i in range(test_data.__len__())]
    result=model.predict_batch(np.array(x))    
    y_prob=np.array(result[0])[:,1]
    test_y=lbl
    fpr, tpr, roc_auc, optimal_threshold, optimal_point=ROC(test_y, y_prob)
    Accuracy,Sensitivity,Specificity=calculate_metric(test_y, y_prob,optimal_threshold)
    result=[optimal_threshold,Accuracy,Sensitivity,Specificity,roc_auc,name]
    results.append(result)
    roc_.append([fpr,tpr,roc_auc,name])

#---------#
import pandas as pd
import matplotlib.pyplot as plt

# 假设 results 和 roc_ 已经定义
df_result = pd.DataFrame(results)
df_result.columns = ["Optimal_threshold", "Accuracy", "Sensitivity", "Specificity", "AUC_ROC", "Model_name"]

# 设置颜色和线宽
colors = ["moccasin", "cornflowerblue", "lightblue", "lightgreen", "yellow", "pink"]
lw = 2

# 创建图形
plt.figure(figsize=(10, 10))

# 绘制 ROC 曲线
for i in range(len(roc_)):
    plt.plot(roc_[i][0], roc_[i][1], color=colors[i], lw=lw, 
             label=f'{roc_[i][3]} (AUC = {roc_[i][2]:0.3f})')

# 绘制对角线
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label='Random Guess')

# 设置图形属性
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=18, labelpad=10)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=18, labelpad=10)
plt.title('ROC Curve', fontsize=20, pad=20)

# 设置刻度字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 设置图例
plt.legend(loc="lower right", fontsize=14, frameon=True, shadow=True, edgecolor='black')

# 保存图形
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')

# 显示图形
plt.show()

#---------#
df_result=pd.DataFrame(results)
df_result.columns=["Optimal_threshold","Accuracy","Sensitivity","Specificity","AUC_ROC","Model_name"]
color=["moccasin","cornflowerblue","lightblue","lightgreen","yellow","pink"]
plt.figure()
plt.figure(figsize=(10,10))
lw = 2
plt.plot(roc_[4][0], roc_[4][1], color=color[4], lw=lw, label= roc_[4][3]+' (AUC = %0.3f)' %  0.941)
plt.plot(roc_[0][0], roc_[0][1], color=color[0], lw=lw, label=roc_[0][3]+' (AUC = %0.3f)' % 0.821) 
plt.plot(roc_[1][0], roc_[1][1], color=color[1], lw=lw, label=roc_[1][3]+' (AUC = %0.3f)' % 0.820) 
plt.plot(roc_[2][0], roc_[2][1], color=color[2], lw=lw, label=roc_[2][3]+' (AUC = %0.3f)' % 0.760) 
plt.plot(roc_[3][0], roc_[3][1], color=color[3], lw=lw, label=roc_[3][3]+' (AUC = %0.3f)' % 0.851)
 
plt.plot(roc_[5][0], roc_[5][1], color=color[5], lw=lw, label=roc_[5][3]+' (AUC = %0.3f)' % 0.650) 
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1-Specificity)',fontsize=18)
plt.ylabel('True Positive Rate (Specificity)',fontsize=18)
plt.title('ROC Curve',fontsize=20)
plt.legend(loc="lower right",frameon=False,fontsize=14)
plt.savefig("roc_curve.png",dpi=300)
plt.show()

#---------#
# for randomstate in range(10000):
data=pd.read_csv('logfourupsample.csv',sep=',')
# 整理数据集，拆分测试集训练集
x, y = data.iloc[:, :-1], data.iloc[:,-1]
train_x, test_x, train_y, test_y = \
    ms.train_test_split(x, y, test_size=0.2, random_state=41)
testdata=pd.concat([test_x,test_y],axis=1)
data_np=np.array(testdata).astype('float32')
selfdata=[]
for i in range(data_np.shape[0]):
    input_np=data_np[i, :-1].reshape([-1,160])
    label_np=data_np[i, -1].astype('int64')
    selfdata.append([input_np,label_np])
testdata=selfdata
# 自定义数据集
class MyDataset(Dataset):
    def __init__(self,mode='train'):
        super(MyDataset, self).__init__()

        if mode == "test":
            self.data =testdata
    def __getitem__(self, index):
            data = self.data[index][0]
            label = self.data[index][1]
            return data, label
    def __len__(self):
        return  len(self.data)
test_data=MyDataset(mode='test')
classes = np.unique(test_y)

precision_list= []
recall_list = []
average_precision_list=[]
for name,model in models:
    clf=model.fit(train_x,train_y)
    pred_proba = clf.predict_proba(test_x)
    probas=pred_proba
    y_true=test_y
    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve( y_true, probas[:, i])

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i], probas[:, i])

# Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)

    precision[micro_key], recall[micro_key], _ = precision_recall_curve(
    y_true.ravel(), probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas,
                                                        average='micro')

    precision_list.append(precision)
    recall_list.append(recall)
    average_precision_list.append(average_precision)

for name,model in cnn_models:   
    x=[test_data.__getitem__(i)[0] for i in range(test_data.__len__())]
    lbl=[test_data.__getitem__(i)[1] for i in range(test_data.__len__())]
    result=model.predict_batch(np.array(x))    
    probas=np.array(result[0])
    y_true=lbl
    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve( y_true, probas[:, i], pos_label=classes[i])
    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))
    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i], probas[:, i])
# Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)
    precision[micro_key], recall[micro_key], _ = precision_recall_curve(
    y_true.ravel(), probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas,
                                                        average='micro')

    precision_list.append(precision)
    recall_list.append(recall)
    average_precision_list.append(average_precision)   

model_list= ['SVM','RandomForest','AdaBoost', 'XGBoost','AMU','CNN']
plt.figure(figsize=(10,10))  
plt.title('Average PR Curve',fontsize=20) 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall',fontsize=18)
plt.ylabel('Precision',fontsize=18)    

# for i in range(6):
#     plt.plot(recall_list[i][micro_key], precision_list[i][micro_key],
#     label=model_list[i]+'(mAP = %0.3f)'%(average_precision_list[i][micro_key]),
#             color=color[i], linewidth=2  )
lw = 2
for i in (4,0,1,2,3,5):
    plt.plot(recall_list[i][micro_key],precision_list[i][micro_key] ,color=color[i], lw=lw, label= model_list[i]+'(mAP = %0.3f)'%(average_precision_list[i][micro_key]))
plt.legend(loc="lower right",frameon=False,fontsize=14)
plt.show
plt.savefig('average_PR_curve_micro41',dpi=300)
    # a=[]
    # ra=[]
    # for i in range(6):      
    #     a.append(average_precision_list[i][micro_key])
    #     if np.argmax(a)==4:            
    #         print('=========================================================================================================')
    #         print(randomstate)
for i in range(6):      
    print(model_list[i],average_precision_list[i][micro_key])

#---------#
#读取外部测试集数据
postdata=pd.read_csv('GSE91061_lognormTPM_post.csv',sep=',')
postdata=np.array(postdata).astype('float32')
preddata=[]
for i in range(postdata.shape[0]):
    input_np=postdata[i, :-1].reshape([-1,160])
    label_np=postdata[i, -1].astype('int64')
    preddata.append([input_np,label_np])
# 自定义数据集
class MyDataset(Dataset):
    def __init__(self,mode='train'):
        super(MyDataset, self).__init__()
        if mode == "pred":
            self.data =preddata
    def __getitem__(self, index):
            data = self.data[index][0]
            label = self.data[index][1]
            return data, label
    def __len__(self):
        return  len(self.data)
pred_data=MyDataset(mode='pred')
classes = np.unique(test_y)

precision_list= []
recall_list = []
average_precision_list=[]
test_x=postdata[:,:-1]
test_y=postdata[:,-1]
for name,model in models:
    clf=model.fit(train_x,train_y)
    probas= clf.predict_proba(test_x)
    y_true=test_y
    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve( y_true, probas[:, i])
    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))
    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i], probas[:, i])
        
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)
    precision[micro_key], recall[micro_key], _ = precision_recall_curve(
    y_true.ravel(), probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas,
                                                        average='micro')

    precision_list.append(precision)
    recall_list.append(recall)
    average_precision_list.append(average_precision)

for name,model in cnn_models:   
    x=[pred_data.__getitem__(i)[0] for i in range(pred_data.__len__())]
    lbl=[pred_data.__getitem__(i)[1] for i in range(pred_data.__len__())]
    result=model.predict_batch(np.array(x))    
    probas=np.array(result[0])
    y_true=lbl
    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve( y_true, probas[:, i], pos_label=classes[i])
    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))
    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i], probas[:, i])
# Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)
    precision[micro_key], recall[micro_key], _ = precision_recall_curve(
    y_true.ravel(), probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas,
                                                        average='micro')

    precision_list.append(precision)
    recall_list.append(recall)
    average_precision_list.append(average_precision)   

model_list= ['SVM','RandomForest','AdaBoost', 'XGBoost','AMU','CNN']
plt.figure(figsize=(10,10))  
plt.title('Average PR Curve in testing dataset',fontsize=20) 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall',fontsize=18)
plt.ylabel('Precision',fontsize=18)    
lw = 2
for i in (4,0,1,2,3,5):
    plt.plot(recall_list[i][micro_key],precision_list[i][micro_key] ,color=color[i], lw=lw, label= model_list[i]+'(mAP = %0.3f)'%(average_precision_list[i][micro_key]))
plt.legend(loc="lower right",frameon=False,fontsize=14)
plt.show
for i in range(6):      
    print(model_list[i],average_precision_list[i][micro_key])

#---------#
results=[]
roc_=[]
test_x=postdata[:,:-1]
test_y=postdata[:,-1]
for name,model in models:
    clf=model.fit(train_x,train_y)
    pred_proba = clf.predict_proba(test_x)
    pred=np.array([np.argmax(i) for i in np.array(pred_proba )])
    print(sm.classification_report(test_y, pred ))
    y_prob=pred_proba[:,1]
    fpr, tpr, roc_auc, optimal_threshold, optimal_point=ROC(test_y, y_prob)
    Accuracy,Sensitivity,Specificity=calculate_metric(test_y, y_prob,optimal_threshold)
    result=[optimal_threshold,Accuracy,Sensitivity,Specificity,roc_auc,name]
    results.append(result)
    roc_.append([fpr,tpr,roc_auc,name])

for name,model in cnn_models:   
    x=[pred_data.__getitem__(i)[0] for i in range(pred_data.__len__())]
    lbl=[pred_data.__getitem__(i)[1] for i in range(pred_data.__len__())]
    result=model.predict_batch(np.array(x))
    y_prob=np.array(result[0])[:,1]
    fpr, tpr, roc_auc, optimal_threshold, optimal_point=ROC(lbl, y_prob)
    Accuracy,Sensitivity,Specificity=calculate_metric(lbl, y_prob,optimal_threshold)
    result=[optimal_threshold,Accuracy,Sensitivity,Specificity,roc_auc,name]
    results.append(result)
    roc_.append([fpr,tpr,roc_auc,name])
df_result=pd.DataFrame(results)
df_result.columns=["Optimal_threshold","Accuracy","Sensitivity","Specificity","AUC_ROC","Model_name"]
 #绘制多组对比roc曲线
color=["moccasin","cornflowerblue","lightblue","lightgreen","yellow","pink"]
plt.figure()
plt.figure(figsize=(10,10))
lw = 2
plt.plot(roc_[4][0], roc_[4][1], color=color[4], lw=lw, label= roc_[4][3]+' (AUC = %0.3f)' %  roc_[4][2])
plt.plot(roc_[0][0], roc_[0][1], color=color[0], lw=lw, label=roc_[0][3]+' (AUC = %0.3f)' % roc_[0][2]) 
plt.plot(roc_[1][0], roc_[1][1], color=color[1], lw=lw, label=roc_[1][3]+' (AUC = %0.3f)' % roc_[1][2]) 
plt.plot(roc_[2][0], roc_[2][1], color=color[2], lw=lw, label=roc_[2][3]+' (AUC = %0.3f)' % roc_[2][2]) 
plt.plot(roc_[3][0], roc_[3][1], color=color[3], lw=lw, label=roc_[3][3]+' (AUC = %0.3f)' % roc_[3][2])
 
plt.plot(roc_[5][0], roc_[5][1], color=color[5], lw=lw, label=roc_[5][3]+' (AUC = %0.3f)' % roc_[5][2]) 
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1-Specificity)',fontsize=18)
plt.ylabel('True Positive Rate (Specificity)',fontsize=18)
plt.title('ROC Curve in testing dataset',fontsize=20)
plt.legend(loc="lower right",frameon=False,fontsize=14)
plt.savefig("roc_curve_testing dataset.png",dpi=300)
plt.show()

#-----------#