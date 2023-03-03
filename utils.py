import pandas as pd
import torch
import json
from torch.utils.data import TensorDataset
class EvalObject(object):
    def __init__(self):
        self.class_accuracy = 0    
        self.class_precision = 0     
        self.class_recall= 0    
        self.class_f1 = 0
        self.tags_accuracy = 0    
        self.tags_precision = 0     
        self.tags_recall= 0    
        self.tags_f1 = 0 
             
def generatedata(filepath, savepath):
    text = []
    labels = []
    with open(filepath, encoding="utf-8") as f:

            s = ""
            label = ""
            test1 = 0
            test2 = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    continue
                else:
                    try:
                      splits = line.split(" ")
                      s+=" "+splits[0]
                      label+=" "+splits[1].strip("\n")
                    except: continue
                if line.startswith("."):
                   text.append(s.strip())
                   labels.append(label.strip())
                   s = ""
                   label = ""
    result = pd.DataFrame({
      "text":text,
      "labels": labels})
    result.to_csv(savepath, sep =",")

train_file = r"/Users/maitrongkhang/Library/CloudStorage/GoogleDrive-khangmt@uit.edu.vn/Other computers/My Laptop/Doctoral program/Lab/NER dataset/DNRTI/train.txt"
save_train =r"/Users/maitrongkhang/Library/CloudStorage/GoogleDrive-khangmt@uit.edu.vn/My Drive/Lab working/Knowledge_from_text/BERT-like/BERT-BiLSTM-CRF/input/DNRTI_train.csv"
eval_file =r"/Users/maitrongkhang/Library/CloudStorage/GoogleDrive-khangmt@uit.edu.vn/Other computers/My Laptop/Doctoral program/Lab/NER dataset/DNRTI/valid.txt"
save_eval= r"/Users/maitrongkhang/Library/CloudStorage/GoogleDrive-khangmt@uit.edu.vn/My Drive/Lab working/Knowledge_from_text/BERT-like/BERT-BiLSTM-CRF/input/DNRTI_eval.csv"

#generatedata(eval_file, save_eval)
class InputFeatures(object):
    # """A single set of features of data."""
    def __init__(self, input_ids, segment_ids, input_mask,  tag_ids, class_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tag_ids = tag_ids
        self.class_ids = class_ids
def align_labels_with_tokens(labels, word_ids, tag2id):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            
            label = tag2id["O"] if word_id is None else tag2id[labels[word_id]]
            
            new_labels.append(label)

            
        elif word_id is None:
            # Special token
            new_labels.append(tag2id["O"] )# we have to treat SEP and CLS as "O" here because CRF do not use default cross entropy loss, so it wont ignore -100. would cause out of range 
        else:
            #same word as previous tokens
            label = tag2id[labels[word_id]]
            # If the label is B-XXX we change it to I-XXX
            if labels[word_id].startswith("B-"):
                temp = "I-" + labels[word_id][2:]
                label = tag2id[temp]
            
            new_labels.append(label)
    assert len(word_ids) == len (new_labels)
    assert new_labels is not None
    for n in new_labels:
        if n > len(tag2id):
            print("error")
            print(n)
    return new_labels
def get_feature_from_sentence(tokenizer, text ,tag_list, class_id, tag2id, num_class =2, maxlength = 512):
    #text is a list of tokens
    #taglist is a list of tags
    #word-piece tokenization and alignment labels
    labels = tag_list
    
    tokenized_inputs = tokenizer(text, truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()
    labels = align_labels_with_tokens(labels, word_ids= word_ids, tag2id = tag2id)
    assert labels is not None
    assert len(tokenized_inputs["input_ids"]) == len(labels)
    if len(tokenized_inputs["input_ids"]) < maxlength:
        padding_length = maxlength - len(tokenized_inputs["input_ids"])
        tokenized_inputs["input_ids"].extend([0]*padding_length)
        tokenized_inputs["token_type_ids"].extend([0]*padding_length)
        tokenized_inputs["attention_mask"].extend([0]*padding_length)
        labels.extend([-100]* padding_length )
    
    
    # assert input_ids is None
    return InputFeatures(tokenized_inputs["input_ids"], tokenized_inputs["token_type_ids"],tokenized_inputs["attention_mask"], tag_ids= labels, class_ids= class_id)

def get_features_from_files(tokenizer, filepath, tag2id):
    features = []
    raw_data = pd.read_csv(filepath, sep=",")      
    tag_lb = [i.split() for i in raw_data['tags'].values.tolist()]
    class_lb = [int(i) for i in raw_data['classes'].values.tolist()]  
    txts = raw_data['text'].values.tolist()
    for text, tag_list, class_id in zip (txts,tag_lb,class_lb):
        feature = get_feature_from_sentence(tokenizer,text,tag_list,class_id,tag2id= tag2id)
        features.append(feature)
    return features
def get_Dataset( tokenizer, file_path, tag2id):
    features = get_features_from_files(tokenizer,file_path, tag2id= tag2id)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_tag_ids = torch.tensor([f.tag_ids for f in features], dtype=torch.long)
    all_class_ids = torch.tensor([f.class_ids for f in features], dtype=torch.long)
    data = DataSequence(all_input_ids,  all_segment_ids, all_input_mask, all_tag_ids, all_class_ids)
    return data
def get_label_map(filepath):
    import pickle
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    tag2id = data["tag2id"]
    id2tag = data["id2tag"]
    num_tags = data["num_tags"]
    return tag2id, id2tag, num_tags

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, input_ids, segment_ids, mask_ids, labels):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.mask_ids = mask_ids
        self.labels = labels
    
    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.input_ids[idx],self.segment_ids[idx],self.mask_ids[idx],self.labels[idx]