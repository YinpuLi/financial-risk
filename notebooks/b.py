import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import pandas as pd
# 加载预训练的BERT模型和分词器
# ./transformers/bert-base-chinese  bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('./transformers/bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./transformers/bert-base-uncased', num_labels=2)

#df = pd.read_csv("./IMDB-Dataset.csv")
def truncate_text(text, max_length=512):
    if len(text) > max_length:
        text = text[:max_length]
    return text
def convert2num(value):
    if value == 'positive':
        return 1
    else:
        return 0
#df['sentiment'] = df['sentiment'].apply(convert2num)
#df['review'] = df['review'].apply(truncate_text)
##train = df[:450]
#test = df[450: 499]
# 准备数据
texts = ["我喜欢打篮球", "这个相机很好看", "今天玩的特别开心", "我不喜欢你", "太糟糕了", "真是件令人伤心的事情"]  # 输入文本
labels = [1, 1, 1, 0, 0, 0]   # 对应的标签，1表示正面，0表示负面
#texts = train['review'].tolist()
#labels = train['sentiment'].tolist()
# 对文本进行分词和编码
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
token_type_ids = inputs['token_type_ids']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 构建DataLoader
dataset = TensorDataset(input_ids, attention_mask, token_type_ids, torch.tensor(labels))
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=32)
cycle_time = 0
# 微调模型
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(10):  # 迭代10个epoch
    total_loss = 0
    for batch in dataloader:
        print("cycle time = ", cycle_time)
        cycle_time = cycle_time+1
        batch = tuple(t.to(device) for t in batch)  # 将batch中的数据移到GPU上
        b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch
        outputs = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        optimizer.zero_grad()  # 清空梯度
    print("Epoch {}/{} - Loss: {:.4f}".format(epoch + 1, 10, total_loss / len(dataloader)))

# 在验证集上评估模型性能

model.eval()
eval_texts = ["我讨厌下雨.", "我不喜欢医院.", "今天的心情真好"]  # 验证集文本
eval_labels = [0, 0, 1]  # 验证集标签
eval_inputs = tokenizer(eval_texts, padding=True, truncation=True, return_tensors='pt')
eval_input_ids = eval_inputs['input_ids']
eval_attention_mask = eval_inputs['attention_mask']
eval_token_type_ids = eval_inputs['token_type_ids']
eval_dataset = TensorDataset(eval_input_ids, eval_attention_mask, eval_token_type_ids, torch.tensor(eval_labels))
eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=2)
eval_loss = 0
eval_accuracy = 0
preds = []
trues = []
with torch.no_grad():
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)  # 将batch中的数据移到GPU上
        b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch

        outputs = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)
        logits = outputs[0]  # 获取最大概率对应的类别标签
        preds += torch.argmax(logits, dim=1).tolist()
        trues += b_labels.tolist()
      # 计算准确率
    print("实际结果：", trues)
    print("预测结果：", preds)

"""
# 加载模型和分词器
#model = torch.load('model.pt')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 要预测的文本
texts = ["你今天看上去不是很开心.", "今天的情况非常糟糕."]

# 对文本进行分词和编码
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
token_type_ids = inputs['token_type_ids']

# 将输入数据移到GPU上
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
token_type_ids = token_type_ids.to(device)

# 使用模型进行预测
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    logits = outputs.logits.argmax(-1)  # 获取最大概率对应的类别标签

# 输出预测结果
print(logits)
"""