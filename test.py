#%%
import pandas as pd
from transformers import BertTokenizerFast
from itertools import repeat
#%%
def tokenize(tokenizer, text,label_list, tag2id , maxlength = 512):
    texts = text.split()
    print(texts)
    labels = label_list
    input_ids = list()
    input_ids.append(101)
    segment_ids = list ()
    segment_ids.append(0)
    attention_mask = list()
    attention_mask.append(1)
    label_ids = list()
    label_ids.append(tag2id["O"])
    for i in range(0,len(texts)):
        split = tokenizer(texts[i],padding= False, max_length = 30, truncation=True)
        temp = [i for i in split["input_ids"] if i not in [101,102]]
        input_ids.extend(temp)
        segment_ids.extend([0]*len(temp))
        attention_mask.extend([1]*len(temp))
        if labels[i].startswith("I-") or labels[i].startswith("O"):
            label_ids.extend([tag2id[labels[i]]]*len(temp))
        if labels[i].startswith("B-"):
            label_ids.append(tag2id[labels[i]])
            newlabel = "I-" + labels[i][2:]
            label_ids.extend([tag2id[newlabel]]*(len(temp)-1))
    input_ids.append(102)
    label_ids.append(tag2id["O"])
    segment_ids.append(0)
    attention_mask.append(1)
    padding_len = maxlength - len(input_ids)
    input_ids.extend([0]* padding_len )
    label_ids.extend([-1]* padding_len )
    segment_ids.extend([0]* padding_len )
    attention_mask.extend([0]* padding_len )
    return input_ids, segment_ids, attention_mask, label_ids

#%%
train_path = r"/Users/maitrongkhang/Library/CloudStorage/GoogleDrive-khangmt@uit.edu.vn/My Drive/Lab working/Knowledge_from_text/BERT-like/BERT-BiLSTM-CRF/input/train_file.csv"
raw_data = pd.read_csv(train_path, sep=",")        
tag_lb = [i.split() for i in raw_data['tags'].values.tolist()]
class_lb = [int(i) for i in raw_data['classes'].values.tolist()]
unique_classes = list ()
        
for lb in class_lb :
    if lb not in unique_classes:
        unique_classes.append(lb)
num_classes = len(unique_classes)

unique_tags= list()
for lb in tag_lb:
    [unique_tags.append(i) for i in lb if i not in unique_tags]
num_tags = len(unique_tags)
        
tag2id = {k: v for v, k in enumerate(unique_tags)}
id2tag = {v: k for v, k in enumerate(unique_tags)}

tags_indexes = [i for i in range(0,len(unique_tags))]
tempdata = pd.DataFrame.from_dict({"tags":unique_tags, "ids": tags_indexes})
tempdata.to_csv(r"/Users/maitrongkhang/Library/CloudStorage/GoogleDrive-khangmt@uit.edu.vn/My Drive/Lab working/Knowledge_from_text/BERT-like/BERT-BiLSTM-CRF/input/tags.csv")

txt = raw_data['text'].values.tolist()


#%%
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
input_ids, segment_ids, attention_mask, label_ids = tokenize(tokenizer, txt[6],tag_lb[6],tag2id,maxlength=512) 
print(len(input_ids))
print(len(label_ids))
data = tokenizer(txt[6],padding= False, max_length = 100, truncation=True)
words = tokenizer.convert_ids_to_tokens(data["input_ids"])
test_label = list()
print(words)
for i in label_ids:
    if i == -1:
        test_label.append("[PAD]")
        continue
    test_label.append(id2tag[i])
print(test_label)
# %%
#create mapping file <tags> and <id>
import pandas as pd
file = r"/Users/maitrongkhang/Library/CloudStorage/GoogleDrive-khangmt@uit.edu.vn/Other computers/My Laptop/Doctoral program/Lab/NER dataset/DNRTI/train.txt"
save = r"/Users/maitrongkhang/Documents/NER/input/mapper.pickle"
unique_tags= list()
with open(file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for l in lines:
        split = l.split()
        if len(split)<2: continue
        if split[1] not in unique_tags:
            unique_tags.append(split[1])

for t in unique_tags:
    if t.startswith("B-"):
        newlabel = "I-" + t[2:]
        if newlabel in unique_tags:
            continue
        else:
            unique_tags.append(newlabel)
import pickle
num_tags = len(unique_tags)
tag2id = {k: v for v, k in enumerate(unique_tags)}
id2tag = {v: k for v, k in enumerate(unique_tags)} 
print(tag2id)
print(id2tag)
data ={"tag2id": tag2id,"id2tag":id2tag, "num_tags": num_tags}
with  open(save,"wb") as f:
    pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)

with open(save,"rb") as f:
    data = pickle.load(f)
    print(data)
# %%
