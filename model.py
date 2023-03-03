import torch
import torch.nn as nn
import torch.functional as F
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel
class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, need_birnn=True,num_tags =5, num_classes =2, rnn_dim=128, ):
        super(BERT_BiLSTM_CRF, self).__init__(config)
        self.loss_function = nn.CrossEntropyLoss() 
        self.num_tags = num_tags #how many tags (B-xx, I-xx, O) are used for NER
        self.num_classes = num_classes #Sentence classification
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.need_birnn = need_birnn

        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim*2
        
        self.hidden2tag = nn.Linear(out_dim, self.num_tags )
        # Plus 1 for -1 tag due to padding
        self.second_head = CRF(self.num_tags , batch_first=True)
        self.firsthead =nn.Linear(config.hidden_size, self.num_classes)
    

    def forward(self, input_ids, tag_ids, class_ids , token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        #output[1] has shape [batch,768], sentence embedding
        output_head1 = outputs[1]
        output_head1 = self.dropout(output_head1)
        output_head1 = self.firsthead(output_head1)

        loss1 = self.loss_function(output_head1, class_ids)
        
        #output[0] has shape [batch,512,768], 512 is the length of padded sentence
        output_head2 = outputs[0]
        if self.need_birnn:
            output_head2, _ = self.birnn(output_head2) # [batch, 512, 256]
        output_head2 = self.dropout(output_head2)
        emissions = self.hidden2tag(output_head2)# [batch,512,num_tags]
        loss2 = -1*self.second_head(emissions, tag_ids, mask=input_mask.byte())
        loss = loss1 + loss2
        return loss

    
    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        output_head1 = outputs[1]
        output_head1 = self.firsthead(output_head1)
        output_head1 = self.dropout(output_head1)
        output_head2 = outputs[0]
        if self.need_birnn:
            output_head2, _ = self.birnn(output_head2)

        output_head2 = self.dropout(output_head2)
        emissions = self.hidden2tag(output_head2)
        
        return {"classes": output_head1,"tags": self.second_head.decode(emissions, input_mask.byte())}