import torch
import torch.nn as nn
import torchvision
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification
import os

class Resnet_Encoder(nn.Module):
    def __init__(self):
        super(Resnet_Encoder, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.model(x)
        # out = self.transion_layer(out)
        # out = self.pool_layer(out.to('cuda:1'))
        out = out.view(out.size(0),-1)
        return out  # BxNx2048

class Bert_lstm(nn.Module):
    def __init__(self,hidden_dim, bert_path, num_layers, dropout):
        super(Bert_lstm,self).__init__()
        #-------------------bert with fine-tuning-------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path,config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True   #true or false
        embedding_dim = self.bert.config.hidden_size
        #-------------------------------------------------------------
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bias=True, batch_first=True,
                              bidirectional=True)
    def forward(self,input_ids, attention_masks):
        # print('layers:!!!!!!!!!!!!!')
        # print('inputs_ids:',input_ids.size(),'attention_masks:',attention_masks.size())
        # print('result:',self.bert(input_ids, attention_mask=attention_masks))
        output = self.bert(input_ids, attention_mask=attention_masks,output_attentions=True,output_hidden_states=True)
        # print('output:',output)
        output = output[0]
        # print(len(self.bilstm(output)))
        out,_ = self.bilstm(output)
        out = out[:,-1,:]
        return out