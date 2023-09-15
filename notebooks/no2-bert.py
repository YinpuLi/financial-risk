#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '/home/aipf/work/userpackages')


# In[2]:


from transformers import BertTokenizer, BertForSequenceClassification
from transformers import  get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from sklearn.metrics import f1_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ------------: ", device)


# In[3]:


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import time
from  tqdm import tqdm
import pandas as pd


# In[4]:


# 特征工程预处理
def truncate_text(text, max_length=500):
    text = text.replace('\\n', '')
    text = text.replace('\n', '')
    text = text.replace('*', '')
    text = text.replace(' ', '')
    text = text.replace('\\', '')
    if len(text) > max_length:
        text = text[:max_length]
    return text
def preprocess(data, lshuffle=False, lbatch_size=4):
    content = data['content'].tolist()
    label = data['label'].tolist()
    # pt represents PyTorch
    inputs = tokenizer(content, padding=True, truncation=True, return_tensors='pt')
    data_set = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], torch.tensor(label))
    data_loader = DataLoader(data_set, batch_size=lbatch_size, shuffle=lshuffle)
    return data_loader


# In[20]:


# 加载预训练的BERT模型和分词器
# ./transformers/bert-base-chinese  bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('/home/aipf/work/团队共享目录/no2/BERT/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('/home/aipf/work/团队共享目录/no2/BERT/bert-base-chinese', num_labels=2)
tqdm.pandas()
df = pd.read_csv("/home/aipf/work/第三届建行杯/赛题/02风险判断/2-赛题数据/train/train.csv")
df['content'] = df['content'].progress_apply(lambda x: x[2:-1])
df['content'] = df['content'].progress_apply(truncate_text)


# In[21]:


y = df['label']
train_tmp, test_tmp, y_train, y_test  = train_test_split(df, y,test_size=0.3, random_state=42, stratify=y)


# In[22]:


train = train_tmp.iloc[0:]
test = test_tmp.iloc[0:]
lbatch_size = 4
epochs = 10


# In[24]:


test.shape


# In[8]:


# 将数据集转换为 BERT输入
# 设置训练参数和优化器

learning_rate = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
warm_up_ratio = 0.1 #定义要预热的step
# 训练的数据
train_loader = preprocess(train, lshuffle=True)
total_steps = (len(train_loader))*epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio*total_steps, num_training_steps=total_steps)


# In[9]:


test.label.value_counts()
val_dataloader = preprocess(test, lshuffle=False)


# In[10]:



test_content = test['content'].tolist()
result = test['label'].tolist()
inputs = tokenizer(test_content, padding=True, truncation=True, return_tensors='pt')
#data_set = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
#data_loader = DataLoader(data_set, batch_size=4, shuffle=False)


# In[11]:



# 微调模型
start = time.time()
best_f1 = 0.0
for epoch in range(epochs):  # 
    model.train()
    total_loss = 0
    epoch_iterator = tqdm(train_loader, desc="Training ( Epoch %d )" %epoch)
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)  # 将batch中的数据移到GPU上
        model = model.to(device)
        b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch
        outputs = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        scheduler.step()
        optimizer.zero_grad()  # 清空梯度
    print("Epoch {}/{} - Loss: {:.4f}".format(epoch + 1, 10, total_loss / len(train_loader)))
    end = time.time()
    runTime = (end - start)
    print("运行时间: ", runTime, "秒")
    
    model.eval()  
    preds = []  
    trues = []
    
    #val_input_ids = val_dataloader[0]
    #val_attention_masks= val_dataloader[1]
    print(len(val_dataloader))
    with torch.no_grad(): 
        epoch_val_iterator = tqdm(val_dataloader, desc="eval ( Epoch %d )" %epoch)
        for step2, batch2 in enumerate(epoch_val_iterator): 
        #for batch2 in val_dataloader:
            val_outputs = None
            val_input_ids = batch2[0].to(device)  
            val_attention_masks = batch2[1].to(device)  
            val_labels = batch2[3].to(device)  
            val_outputs = model(val_input_ids, attention_mask=val_attention_masks)  
            logits = val_outputs[0] 
            #print(logits)
            preds += torch.argmax(logits, dim=1).tolist() 
            #print(preds)
            trues += val_labels.tolist() 
            #print(trues)
        #auc = roc_auc_score(trues, preds)
        #print("AUC: ", auc)
        f1 = f1_score(trues, preds, average='macro') # 计算宏平均的F1 Score，每个类别权重相同
        print(f1)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pth')


# In[15]:


testA_path='/home/aipf/work/第三届建行杯/赛题/02风险判断/2-赛题数据/testA/testA.csv'
testA_data = pd.read_csv(testA_path)
testA_data.shape


# In[16]:


testA_data['content'] = testA_data['content'].progress_apply(lambda x: x[2:-1])
testA_data['content'] = testA_data['content'].progress_apply(truncate_text)
testA_data['label'] = [0]*len(testA_data)


# In[ ]:





# In[17]:


model.eval()  
preds = []    
with torch.no_grad(): 
    epoch_test_iterator = tqdm(preprocess(testA_data), desc="test:")
    for step3, batch3 in enumerate(epoch_test_iterator): 
    #for batch3 in data_loader: 
        # for step, batch in enumerate(epoch_iterator):
        val_input_ids = batch3[0].to(device)  
        val_attention_masks = batch3[1].to(device)  
        #val_labels = batch3[3].to(device)  
        val_outputs = model(val_input_ids, attention_mask=val_attention_masks)  
        logits = val_outputs[0]
        preds += torch.argmax(logits, dim=1).tolist()  


# In[18]:


testA_data['pred'] = preds


# In[19]:


testA_data.pred.value_counts()


# In[28]:


testA_data = testA_data.drop(['content'], axis=1)
testA_data = testA_data.drop(['label'], axis=1)
testA_data = testA_data.set_index('ID')
testA_data.to_csv('/home/aipf/work/第三届建行杯/作品/NO2-Bert-202309141700.csv')