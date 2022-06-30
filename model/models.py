import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable,Function
from layers import Bert_lstm,Resnet_Encoder

from layers import *
from transformers import BertModel, BertConfig
class Scaled_Dot_Product_Attention_pos(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention_pos, self).__init__()

    def forward(self, Q, K, V, scale,kg_sim):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        beta = torch.mul(attention, kg_sim)
        beta = F.softmax(beta,dim = -1)
        # print('beta size:',beta.size()) #128,1,5
        # print('v size:',V.size())#128,5,80
        context = torch.matmul(beta, V)
        # print('v after attention:',context.size()) #128,1,80
        return context

class Scaled_Dot_Product_Attention_neg(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention_neg, self).__init__()

    def forward(self, Q, K, V, scale, kg_sim):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = -1*torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = -1*F.softmax(attention, dim=-1)
        beta = torch.mul(attention, kg_sim)
        beta = F.softmax(beta, dim=-1)
        # print('beta size:', beta.size())  # 128,1,5
        # print('v size:',V.size())#128,5,80
        context = torch.matmul(beta, V)
        # print('v after attention:', context.size())  # 128,1,80
        # context = torch.matmul(attention, V)
        return context

class ReverseLayerF(Function):

    # @staticmethod
    def forward(self, x):
        # self.lambd = args.lambda_m
        return x.view_as(x)

    # @staticmethod
    def backward(self, grad_output):
        return (-grad_output)


def grad_reverse(x):
    return ReverseLayerF()(x)


#initial初始化策略：当没有图片时：将图片初始化embedding替换为文本初始化embedding
#有图片时维持原状
class inconsistency_model_initial(nn.Module):
    def __init__(self,config,pathset):
        #img_hidden_size, bert_path, hidden_dim, num_layers, dropout=0.5
        super(inconsistency_model_initial, self).__init__()
        self.txt_hidden_dim = config.hidden_dim
        self.img_hidden_size = config.img_hidden_size
        self.bert_path = pathset.path_bert
        self.dropout = config.dropout
        self.num_layers = config.num_layers
        # self.clf = nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2,2)
        # self.clf = nn.Sequential(nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2+100, 1),
        #                          nn.Sigmoid())
        # self.dropout = dropout
        self.ln_txt = nn.Linear(self.txt_hidden_dim * 2, 200)
        self.ln_img = nn.Linear(self.img_hidden_size, 200)
        self.ln_shr = nn.Linear(200, 40, bias=False)
        self.ln_uniq_txt = nn.Linear(200, 40, bias=False)
        self.ln_uniq_img = nn.Linear(200, 40, bias=False)
        self.ln_kg1 = nn.Linear(50, 40)
        self.ln_kg2 = nn.Linear(160, 120)

        self.txtenc = Bert_lstm(self.txt_hidden_dim, self.bert_path, self.num_layers, self.dropout)
        self.imgenc = Resnet_Encoder()
        # 损失函数增加unique layer和shared layer
        # ---------------------------------
        #文本替换图片
        self.ln_txt2img = nn.Linear(300,224)


        # 多头attention
        self.num_head = 1
        self.dim_model = 80
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head
        self.fc_Q_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_Q_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.attention_pos = Scaled_Dot_Product_Attention_pos()
        self.attention_neg = Scaled_Dot_Product_Attention_neg()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.dim_model)
        #rumor_classifier
        # self.rumor_classifier = nn.Sequential(nn.Linear(360, 1),
        #                          nn.Sigmoid())
        self.rumor_classifier = nn.Sequential()
        self.rumor_classifier.add_module('r_fc1',nn.Linear(360,180))
        self.rumor_classifier.add_module('r_relu1',nn.LeakyReLU(True))
        self.rumor_classifier.add_module('r_fc2', nn.Linear(180, 1))
        self.rumor_classifier.add_module('r_softmax', nn.Sigmoid())

        #modal_classifier
        self.modal_classifier = nn.Sequential()
        self.modal_classifier.add_module('m_fc1', nn.Linear(40, 20))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.modal_classifier.add_module('m_relu1', nn.LeakyReLU(True))
        self.modal_classifier.add_module('m_fc2', nn.Linear(20, 1))
        self.modal_classifier.add_module('m_softmax', nn.Sigmoid())

    def forward(self, txt_token, txt_masks, img, kg1, kg2, kg_sim, y_img,args):
        # print('model:',kg1.size(),kg2.size(),kg_sim.size())
        #torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 1, 5])
        txt = self.txtenc(txt_token, txt_masks)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # print('txt.size:',txt.size())  #(batch,300)
        # print('img.size:', img.size()) # (batch,3,224,224)

        txt_temp = self.ln_txt2img(txt)  #(batch,224)
        txt_temp = txt_temp.unsqueeze(1) #(batch,1,224)
        # print('1-----txt_temp size:', txt_temp.size())
        txt_temp = txt_temp.expand(-1, 224,-1)
        # print('2-----txt_temp size:', txt_temp.size())
        txt_temp = txt_temp.unsqueeze(1) #(batch ,1,224,224)
        # print('3-----txt_temp size:', txt_temp.size())
        txt_temp = txt_temp.expand(-1,3,-1,-1)
        # print('4-----txt_temp size:',txt_temp)

        img_re = y_img.unsqueeze(1)
        txt_re = torch.sub(1, y_img).unsqueeze(1)
        # print('img_re_before:', img_re)
        # print('txt_re_before:', txt_re)

        img_re = img_re.expand(-1,224)
        txt_re = txt_re.expand(-1,224)

        img_re = img_re.unsqueeze(1)
        txt_re = txt_re.unsqueeze(1)

        img_re = img_re.expand(-1,224,-1)
        txt_re = txt_re.expand(-1,224,-1)

        img_re = img_re.unsqueeze(1)
        txt_re = txt_re.unsqueeze(1)

        img_re = img_re.expand(-1, 3, -1, -1)
        txt_re = txt_re.expand(-1, 3, -1, -1)
        # print('img_re:',img_re)
        # print('txt_re:',txt_re)

        img_new = torch.mul(img_re, img) + torch.mul(txt_re, txt_temp)
        # print('img_new size:',img_new.size())


        img = self.imgenc(img_new)
        # img = torch.dropout(img, self.dropout, train=self.training)
        # print('img size:',img.size())
        # txt = self.ln_txt(txt)
        # txt = self.dropout(txt)
        txt = F.leaky_relu(self.ln_txt(txt))
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # txt = self.dropout(txt)
        # img = self.ln_img(img)
        # img = self.dropout(img)
        img = F.leaky_relu(self.ln_img(img))
        # img = torch.dropout(img, self.dropout, train=self.training)
        # img = self.dropout(img)
        txt_share = self.ln_shr(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_share = self.ln_shr(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        txt_uniq = self.ln_uniq_txt(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_uniq = self.ln_uniq_img(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        modal_shr = torch.cat([txt_share, img_share], -1)  # 80
        # kg1 = self.ln_kg1(kg1)
        # kg1 = self.dropout(kg1)
        kg1 = F.leaky_relu(self.ln_kg1(kg1))
        # kg1 = torch.dropout(kg1, self.dropout, train=self.training)
        # kg2 = self.ln_kg1(kg2)
        # kg2 = self.dropout(kg2)
        kg2 = F.leaky_relu(self.ln_kg1(kg2))
        # kg2 = torch.dropout(kg2, self.dropout, train=self.training)
        # kg1 = self.ln_kg1(kg1)
        cat_kg = torch.cat([kg1, kg2], -1)  # 80
        # print('modal_shr size:',modal_shr.size())
        # print('cat_kg size:',cat_kg.size())
        # Q: modal_shr K:cat_kg V:cat_kg
        # ----------------------------------------------
        batch_size = cat_kg.size(0)
        Q_pos = self.fc_Q_pos(modal_shr)
        K_pos = self.fc_K_pos(cat_kg)
        V_pos = self.fc_V_pos(cat_kg)
        Q_pos = Q_pos.view(batch_size * self.num_head, -1, self.dim_head)
        K_pos = K_pos.view(batch_size * self.num_head, -1, self.dim_head)
        V_pos = V_pos.view(batch_size * self.num_head, -1, self.dim_head)

        Q_neg = self.fc_Q_neg(modal_shr)
        K_neg = self.fc_K_neg(cat_kg)
        V_neg = self.fc_V_neg(cat_kg)
        Q_neg = Q_neg.view(batch_size * self.num_head, -1, self.dim_head)
        K_neg = K_neg.view(batch_size * self.num_head, -1, self.dim_head)
        V_neg = V_neg.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K_pos.size(-1) ** -0.5  # 缩放因子
        kg_context_pos = self.attention_pos(Q_pos, K_pos, V_pos, scale, kg_sim)
        kg_context_pos = kg_context_pos.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_pos = self.fc1(kg_context_pos)
        kg_context_pos = self.dropout(kg_context_pos)
        # kg_context_pos = torch.dropout(kg_context_pos, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_pos = self.layer_norm(kg_context_pos)
        kg_context_pos = kg_context_pos.squeeze(1)

        kg_context_neg = self.attention_neg(Q_neg, K_neg, V_neg, scale, kg_sim)
        kg_context_neg = kg_context_neg.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_neg = self.fc2(kg_context_neg)
        kg_context_neg = self.dropout(kg_context_neg)
        # kg_cintext_neg = torch.dropout(kg_context_neg, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_neg = self.layer_norm(kg_context_neg)
        kg_context_neg = kg_context_neg.squeeze(1)
        # ------------------------------------------------------------------
        # print('kg_context size:',kg_context.size())
        cat_context = torch.cat([kg_context_pos, kg_context_neg], -1)
        # kg_context = self.ln_kg2(cat_context)
        # kg_context = self.dropout(kg_context)
        kg_context = F.leaky_relu(self.ln_kg2(cat_context))  # 120
        # kg_context = torch.dropout(kg_context, self.dropout, train=self.training)
        sub = txt_uniq - img_uniq
        hadmard = torch.mul(txt_share, img_share)
        post_uniq_context = torch.cat([txt_uniq, sub, img_uniq], -1)  # 120
        post_share_context = torch.cat([txt_share, hadmard, img_share], -1)  # 120
        # print('model:',post_share_context.size(),post_uniq_context.size(),kg_context.size())
        cat = torch.cat([post_share_context, post_uniq_context, kg_context], -1)
        # print('cat size:', cat.size())
        output_class = self.rumor_classifier(cat)
        #-----------------调用 grad_reverse-------------------
        reverse_feature = grad_reverse(img_share)
        output_modal = self.modal_classifier(reverse_feature)
        # return F.softmax(out,dim=1)
        return output_class, output_modal

#用门控控制三条路上的输出，增加“是否有图片”标签的输入
class inconsistency_model_weight(nn.Module):
    def __init__(self,config, pathset):
        super(inconsistency_model_weight,self).__init__()
        self.txt_hidden_dim = config.hidden_dim
        self.img_hidden_size = config.img_hidden_size
        self.bert_path = pathset.bert_path
        # self.dropout = dropout
        # self.clf = nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2,2)
        # self.clf = nn.Sequential(nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2+100, 1),
        #                          nn.Sigmoid())
        # self.dropout = dropout
        self.ln_txt = nn.Linear(self.txt_hidden_dim * 2, 200)
        self.ln_img = nn.Linear(self.img_hidden_size, 200)
        self.ln_shr = nn.Linear(200,40,bias=False)
        self.ln_uniq_txt = nn.Linear(200,40,bias=False)
        self.ln_uniq_img = nn.Linear(200,40,bias=False)
        self.ln_kg1 = nn.Linear(50, 40)
        self.ln_kg2 = nn.Linear(160,120)
        self.clf = nn.Sequential(nn.Linear(360, 1),
                                 nn.Sigmoid())
        self.clf2 = nn.Sequential(nn.Linear(120, 1),
                                 nn.Sigmoid())
        self.txtenc = Bert_lstm(self.txt_hidden_dim, self.bert_path, config.num_layers, dropout=config.dropout)
        self.imgenc = Resnet_Encoder()
        #损失函数增加unique layer和shared layer
        #---------------------------------
        #多头attention
        self.num_head = 1
        self.dim_model = 80
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head
        self.fc_Q_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_Q_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.attention_pos = Scaled_Dot_Product_Attention_pos()
        self.attention_neg = Scaled_Dot_Product_Attention_neg()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(self.dim_model)
        #------------------------------------------------------------------
        self.ln_clf = nn.Linear(120, 120)
        self.tanh1 = nn.Tanh()
        self.v_weight = nn.Parameter(torch.zeros(120))

    def forward(self, txt_input_ids, txt_attention_masks, img, kg1, kg2, kg_sim):
        txt = self.txtenc(txt_input_ids, txt_attention_masks)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # print('txt.size:',txt.size())
        img = self.imgenc(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        # print('img size:',img.size())
        txt = self.ln_txt(txt)
        txt = self.dropout(txt)
        txt = F.leaky_relu(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # txt = self.dropout(txt)
        img = self.ln_img(img)
        img = self.dropout(img)
        img = F.leaky_relu(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        # img = self.dropout(img)
        txt_share = self.ln_shr(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_share = self.ln_shr(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        txt_uniq = self.ln_uniq_txt(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_uniq = self.ln_uniq_img(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        modal_shr = torch.cat([txt_share,img_share],-1)#80
        kg1 = F.leaky_relu(self.ln_kg1(kg1))
        # kg1 = torch.dropout(kg1, self.dropout, train=self.training)
        kg2 = F.leaky_relu(self.ln_kg1(kg2))
        # kg2 = torch.dropout(kg2, self.dropout, train=self.training)
        # kg1 = self.ln_kg1(kg1)
        cat_kg = torch.cat([kg1, kg2], -1)  #80
        # print('modal_shr size:',modal_shr.size())
        # print('cat_kg size:',cat_kg.size())
        #Q: modal_shr K:cat_kg V:cat_kg
        #----------------------------------------------
        batch_size = cat_kg.size(0)
        Q_pos = self.fc_Q_pos(modal_shr)
        K_pos = self.fc_K_pos(cat_kg)
        V_pos = self.fc_V_pos(cat_kg)
        Q_pos = Q_pos.view(batch_size * self.num_head, -1, self.dim_head)
        K_pos = K_pos.view(batch_size * self.num_head, -1, self.dim_head)
        V_pos = V_pos.view(batch_size * self.num_head, -1, self.dim_head)

        Q_neg = self.fc_Q_neg(modal_shr)
        K_neg = self.fc_K_neg(cat_kg)
        V_neg = self.fc_V_neg(cat_kg)
        Q_neg = Q_neg.view(batch_size * self.num_head, -1, self.dim_head)
        K_neg = K_neg.view(batch_size * self.num_head, -1, self.dim_head)
        V_neg = V_neg.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K_pos.size(-1) ** -0.5  # 缩放因子
        kg_context_pos = self.attention_pos(Q_pos, K_pos, V_pos, scale,kg_sim)
        kg_context_pos = kg_context_pos.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_pos = self.fc1(kg_context_pos)
        kg_context_pos = self.dropout(kg_context_pos)
        # kg_context_pos = torch.dropout(kg_context_pos, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_pos = self.layer_norm(kg_context_pos)
        kg_context_pos = kg_context_pos.squeeze(1)

        kg_context_neg = self.attention_neg(Q_neg, K_neg, V_neg, scale,kg_sim)
        kg_context_neg = kg_context_neg.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_neg = self.fc2(kg_context_neg)
        kg_context_neg = self.dropout(kg_context_neg)
        # kg_cintext_neg = torch.dropout(kg_context_neg, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_neg = self.layer_norm(kg_context_neg)
        kg_context_neg = kg_context_neg.squeeze(1)
        # ------------------------------------------------------------------
        # print('kg_context size:',kg_context.size())
        cat_context = torch.cat([kg_context_pos,kg_context_neg],-1)
        kg_context = F.leaky_relu(self.ln_kg2(cat_context)) #120
        # kg_context = torch.dropout(kg_context, self.dropout, train=self.training)
        sub = txt_uniq - img_uniq
        hadmard = torch.mul(txt_share,img_share)
        post_uniq_context = torch.cat([txt_uniq,sub,img_uniq],-1) #120
        post_share_context = torch.cat([txt_share,hadmard,img_share],-1) #120

        # cat_kg_h = self.tanh1(self.ln_kg3(cat_kg))  # [64,5,80]
        # alpha = F.softmax(torch.matmul(cat_kg_h, self.w1), dim=1).unsqueeze(-1)
        # cat_kg_a = cat_kg * alpha
        # print('kg_context size',kg_context.size()) #[32,120]
        # print('post_share_context size',post_share_context.size()) #[32,120]
        # print('post uniq context size',post_uniq_context.size()) #[32,120]
        kg_context = kg_context.unsqueeze(1)
        post_share_context = post_share_context.unsqueeze(1)
        post_uniq_context = post_uniq_context.unsqueeze(1)
        cat_all = torch.cat([kg_context,post_share_context,post_uniq_context],dim=1) #[32,3,120]
        cat_all_h = self.tanh1(self.ln_clf(cat_all))
        alpha_clf = F.softmax(torch.matmul(cat_all_h, self.v_weight), dim=1).unsqueeze(-1)
        cat_all_a = cat_all * alpha_clf
        # print('alpha_clf:',alpha_clf)
        # print('alpha_clf size:', alpha_clf.size()) #[32,3,1]
        # print('cat_all_a:',cat_all_a.size()) #[32,3,120]
        # cat = torch.cat([post_share_context, post_uniq_context, kg_context], -1)
        # print('cat size:', cat.size())
        #------------------------------concatenate 1-----------------------------------
        # cat_all_a = cat_all_a.view(cat_all_a.size(0),1,-1)
        # cat_all_a = cat_all_a.squeeze(1)
        # print('concatenate 1 cat_all_a size',cat_all_a.size())
        #-----------------------------concatenate 2 -----------------------------------
        cat_all_a = torch.sum(cat_all_a,dim=1)
        # print('concatenate 2 cat_all_a size', cat_all_a.size())

        out = self.clf2(cat_all_a)
        # return F.softmax(out,dim=1)
        return out

#对抗学习的方法，增加“是否有图片”标签的输入
class inconsistency_model_adversarial(nn.Module):
    def __init__(self,config,pathset):
        #img_hidden_size, bert_path, hidden_dim, num_layers, dropout=0.5
        super(inconsistency_model_adversarial, self).__init__()
        self.txt_hidden_dim = config.hidden_dim
        self.img_hidden_size = config.img_hidden_size
        self.bert_path = pathset.path_bert
        self.dropout = config.dropout
        self.num_layers = config.num_layers
        # self.clf = nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2,2)
        # self.clf = nn.Sequential(nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2+100, 1),
        #                          nn.Sigmoid())
        # self.dropout = dropout
        self.ln_txt = nn.Linear(self.txt_hidden_dim * 2, 200)
        self.ln_img = nn.Linear(self.img_hidden_size, 200)
        self.ln_shr = nn.Linear(200, 40, bias=False)
        self.ln_uniq_txt = nn.Linear(200, 40, bias=False)
        self.ln_uniq_img = nn.Linear(200, 40, bias=False)
        self.ln_kg1 = nn.Linear(50, 40)
        self.ln_kg2 = nn.Linear(160, 120)

        self.txtenc = Bert_lstm(self.txt_hidden_dim, self.bert_path, self.num_layers, self.dropout)
        self.imgenc = Resnet_Encoder()
        # 损失函数增加unique layer和shared layer
        # ---------------------------------
        # 多头attention
        self.num_head = 1
        self.dim_model = 80
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head
        self.fc_Q_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_Q_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.attention_pos = Scaled_Dot_Product_Attention_pos()
        self.attention_neg = Scaled_Dot_Product_Attention_neg()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.dim_model)
        #rumor_classifier
        # self.rumor_classifier = nn.Sequential(nn.Linear(360, 1),
        #                          nn.Sigmoid())
        self.rumor_classifier = nn.Sequential()
        self.rumor_classifier.add_module('r_fc1',nn.Linear(360,180))
        self.rumor_classifier.add_module('r_relu1',nn.LeakyReLU(True))
        self.rumor_classifier.add_module('r_fc2', nn.Linear(180, 1))
        self.rumor_classifier.add_module('r_softmax', nn.Sigmoid())

        #modal_classifier
        self.modal_classifier = nn.Sequential()
        self.modal_classifier.add_module('m_fc1', nn.Linear(40, 20))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.modal_classifier.add_module('m_relu1', nn.LeakyReLU(True))
        self.modal_classifier.add_module('m_fc2', nn.Linear(20, 1))
        self.modal_classifier.add_module('m_softmax', nn.Sigmoid())

    def forward(self, txt_token, txt_masks, img, kg1, kg2, kg_sim,args):
        # print('model:',kg1.size(),kg2.size(),kg_sim.size())
        #torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 1, 5])
        txt = self.txtenc(txt_token, txt_masks)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # print('txt.size:',txt.size())
        img = self.imgenc(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        # print('img size:',img.size())
        # txt = self.ln_txt(txt)
        # txt = self.dropout(txt)
        txt = F.leaky_relu(self.ln_txt(txt))
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # txt = self.dropout(txt)
        # img = self.ln_img(img)
        # img = self.dropout(img)
        img = F.leaky_relu(self.ln_img(img))
        # img = torch.dropout(img, self.dropout, train=self.training)
        # img = self.dropout(img)
        txt_share = self.ln_shr(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_share = self.ln_shr(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        txt_uniq = self.ln_uniq_txt(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_uniq = self.ln_uniq_img(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        modal_shr = torch.cat([txt_share, img_share], -1)  # 80
        # kg1 = self.ln_kg1(kg1)
        # kg1 = self.dropout(kg1)
        kg1 = F.leaky_relu(self.ln_kg1(kg1))
        # kg1 = torch.dropout(kg1, self.dropout, train=self.training)
        # kg2 = self.ln_kg1(kg2)
        # kg2 = self.dropout(kg2)
        kg2 = F.leaky_relu(self.ln_kg1(kg2))
        # kg2 = torch.dropout(kg2, self.dropout, train=self.training)
        # kg1 = self.ln_kg1(kg1)
        cat_kg = torch.cat([kg1, kg2], -1)  # 80
        # print('modal_shr size:',modal_shr.size())
        # print('cat_kg size:',cat_kg.size())
        # Q: modal_shr K:cat_kg V:cat_kg
        # ----------------------------------------------
        batch_size = cat_kg.size(0)
        Q_pos = self.fc_Q_pos(modal_shr)
        K_pos = self.fc_K_pos(cat_kg)
        V_pos = self.fc_V_pos(cat_kg)
        Q_pos = Q_pos.view(batch_size * self.num_head, -1, self.dim_head)
        K_pos = K_pos.view(batch_size * self.num_head, -1, self.dim_head)
        V_pos = V_pos.view(batch_size * self.num_head, -1, self.dim_head)

        Q_neg = self.fc_Q_neg(modal_shr)
        K_neg = self.fc_K_neg(cat_kg)
        V_neg = self.fc_V_neg(cat_kg)
        Q_neg = Q_neg.view(batch_size * self.num_head, -1, self.dim_head)
        K_neg = K_neg.view(batch_size * self.num_head, -1, self.dim_head)
        V_neg = V_neg.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K_pos.size(-1) ** -0.5  # 缩放因子
        kg_context_pos = self.attention_pos(Q_pos, K_pos, V_pos, scale, kg_sim)
        kg_context_pos = kg_context_pos.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_pos = self.fc1(kg_context_pos)
        kg_context_pos = self.dropout(kg_context_pos)
        # kg_context_pos = torch.dropout(kg_context_pos, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_pos = self.layer_norm(kg_context_pos)
        kg_context_pos = kg_context_pos.squeeze(1)

        kg_context_neg = self.attention_neg(Q_neg, K_neg, V_neg, scale, kg_sim)
        kg_context_neg = kg_context_neg.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_neg = self.fc2(kg_context_neg)
        kg_context_neg = self.dropout(kg_context_neg)
        # kg_cintext_neg = torch.dropout(kg_context_neg, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_neg = self.layer_norm(kg_context_neg)
        kg_context_neg = kg_context_neg.squeeze(1)
        # ------------------------------------------------------------------
        # print('kg_context size:',kg_context.size())
        cat_context = torch.cat([kg_context_pos, kg_context_neg], -1)
        # kg_context = self.ln_kg2(cat_context)
        # kg_context = self.dropout(kg_context)
        kg_context = F.leaky_relu(self.ln_kg2(cat_context))  # 120
        # kg_context = torch.dropout(kg_context, self.dropout, train=self.training)
        sub = txt_uniq - img_uniq
        hadmard = torch.mul(txt_share, img_share)
        post_uniq_context = torch.cat([txt_uniq, sub, img_uniq], -1)  # 120
        post_share_context = torch.cat([txt_share, hadmard, img_share], -1)  # 120
        # print('model:',post_share_context.size(),post_uniq_context.size(),kg_context.size())
        cat = torch.cat([post_share_context, post_uniq_context, kg_context], -1)
        # print('cat size:', cat.size())
        output_class = self.rumor_classifier(cat)
        #-----------------调用 grad_reverse-------------------
        reverse_feature = grad_reverse(img_share)
        output_modal = self.modal_classifier(reverse_feature)
        # return F.softmax(out,dim=1)
        return output_class, output_modal

#用文本输入替代没有图片的图片部分输入，增加“是否有图片”标签的输入
#T_u和I_u之间的差用绝对值, T_u和I_u之间的点积用绝对值
class inconsistency_model_replace(nn.Module):
    def __init__(self,config,pathset):
        #img_hidden_size, bert_path, hidden_dim, num_layers, dropout=0.5
        super(inconsistency_model_replace, self).__init__()
        self.txt_hidden_dim = config.hidden_dim
        self.img_hidden_size = config.img_hidden_size
        self.bert_path = pathset.path_bert
        self.dropout = config.dropout
        self.num_layers = config.num_layers
        # self.clf = nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2,2)
        # self.clf = nn.Sequential(nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2+100, 1),
        #                          nn.Sigmoid())
        # self.dropout = dropout
        self.ln_txt = nn.Linear(self.txt_hidden_dim * 2, 200)
        self.ln_img = nn.Linear(self.img_hidden_size, 200)
        self.ln_shr = nn.Linear(200, 40, bias=False)
        self.ln_uniq_txt = nn.Linear(200, 40, bias=False)
        self.ln_uniq_img = nn.Linear(200, 40, bias=False)
        self.ln_kg1 = nn.Linear(50, 40)
        self.ln_kg2 = nn.Linear(160, 120)

        self.txtenc = Bert_lstm(self.txt_hidden_dim, self.bert_path, self.num_layers, self.dropout)
        self.imgenc = Resnet_Encoder()
        # 损失函数增加unique layer和shared layer
        # ---------------------------------
        # 多头attention
        self.num_head = 1
        self.dim_model = 80
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head
        self.fc_Q_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_Q_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.attention_pos = Scaled_Dot_Product_Attention_pos()
        self.attention_neg = Scaled_Dot_Product_Attention_neg()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.dim_model)
        #rumor_classifier
        # self.rumor_classifier = nn.Sequential(nn.Linear(360, 1),
        #                          nn.Sigmoid())
        self.rumor_classifier = nn.Sequential()
        self.rumor_classifier.add_module('r_fc1',nn.Linear(360,180))
        self.rumor_classifier.add_module('r_relu1',nn.LeakyReLU(True))
        self.rumor_classifier.add_module('r_fc2', nn.Linear(180, 1))
        self.rumor_classifier.add_module('r_softmax', nn.Sigmoid())

        #modal_classifier
        self.modal_classifier = nn.Sequential()
        self.modal_classifier.add_module('m_fc1', nn.Linear(40, 20))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.modal_classifier.add_module('m_relu1', nn.LeakyReLU(True))
        self.modal_classifier.add_module('m_fc2', nn.Linear(20, 1))
        self.modal_classifier.add_module('m_softmax', nn.Sigmoid())

    def forward(self, txt_token, txt_masks, img, kg1, kg2, kg_sim, y_img, args):
        # print('model:',kg1.size(),kg2.size(),kg_sim.size())
        #torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 1, 5])
        txt = self.txtenc(txt_token, txt_masks)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # print('txt.size:',txt.size())
        img = self.imgenc(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        # print('img size:',img.size())
        # txt = self.ln_txt(txt)
        # txt = self.dropout(txt)
        txt = F.leaky_relu(self.ln_txt(txt))
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # txt = self.dropout(txt)
        # img = self.ln_img(img)
        # img = self.dropout(img)
        img = F.leaky_relu(self.ln_img(img))
        # img = torch.dropout(img, self.dropout, train=self.training)
        # img = self.dropout(img)
        txt_share = self.ln_shr(txt)
        # print('y_img size:',y_img.size()) #torch.Size([32])
        # print('txt_share size:',txt_share.size())#torch.Size([32,40])
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_share = self.ln_shr(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        txt_uniq = self.ln_uniq_txt(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_uniq = self.ln_uniq_img(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        #图片和文本reserved
        img_re = y_img.unsqueeze(1)
        txt_re = torch.sub(1,y_img).unsqueeze(1)
        # print('img_re_before:', img_re)
        # print('txt_re_before:', txt_re)

        img_re = img_re.expand(-1,40)
        txt_re = txt_re.expand(-1,40)

        # print('img_re:',img_re[0])
        # print('txt_re:',txt_re[0])
        img_share_new = torch.mul(img_re,img_share)+torch.mul(txt_re,txt_share)
        img_uniq_new = torch.mul(img_re,img_uniq)+torch.mul(txt_re,txt_uniq)
        #
        # print('img_re:',img_re)
        # print('img_share:',img_share[0])
        # print('txt_share:',txt_share[0])
        # print('img_share_new:',img_share_new[0])

        modal_shr = torch.cat([txt_share, img_share_new], -1)  # 80
        # kg1 = self.ln_kg1(kg1)
        # kg1 = self.dropout(kg1)
        kg1 = F.leaky_relu(self.ln_kg1(kg1))
        # kg1 = torch.dropout(kg1, self.dropout, train=self.training)
        # kg2 = self.ln_kg1(kg2)
        # kg2 = self.dropout(kg2)
        kg2 = F.leaky_relu(self.ln_kg1(kg2))
        # kg2 = torch.dropout(kg2, self.dropout, train=self.training)
        # kg1 = self.ln_kg1(kg1)
        cat_kg = torch.cat([kg1, kg2], -1)  # 80
        # print('modal_shr size:',modal_shr.size())
        # print('cat_kg size:',cat_kg.size())
        # Q: modal_shr K:cat_kg V:cat_kg
        # ----------------------------------------------
        batch_size = cat_kg.size(0)
        Q_pos = self.fc_Q_pos(modal_shr)
        K_pos = self.fc_K_pos(cat_kg)
        V_pos = self.fc_V_pos(cat_kg)
        Q_pos = Q_pos.view(batch_size * self.num_head, -1, self.dim_head)
        K_pos = K_pos.view(batch_size * self.num_head, -1, self.dim_head)
        V_pos = V_pos.view(batch_size * self.num_head, -1, self.dim_head)

        Q_neg = self.fc_Q_neg(modal_shr)
        K_neg = self.fc_K_neg(cat_kg)
        V_neg = self.fc_V_neg(cat_kg)
        Q_neg = Q_neg.view(batch_size * self.num_head, -1, self.dim_head)
        K_neg = K_neg.view(batch_size * self.num_head, -1, self.dim_head)
        V_neg = V_neg.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K_pos.size(-1) ** -0.5  # 缩放因子
        kg_context_pos = self.attention_pos(Q_pos, K_pos, V_pos, scale, kg_sim)
        kg_context_pos = kg_context_pos.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_pos = self.fc1(kg_context_pos)
        kg_context_pos = self.dropout(kg_context_pos)
        # kg_context_pos = torch.dropout(kg_context_pos, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_pos = self.layer_norm(kg_context_pos)
        kg_context_pos = kg_context_pos.squeeze(1)

        kg_context_neg = self.attention_neg(Q_neg, K_neg, V_neg, scale, kg_sim)
        kg_context_neg = kg_context_neg.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_neg = self.fc2(kg_context_neg)
        kg_context_neg = self.dropout(kg_context_neg)
        # kg_cintext_neg = torch.dropout(kg_context_neg, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_neg = self.layer_norm(kg_context_neg)
        kg_context_neg = kg_context_neg.squeeze(1)
        # ------------------------------------------------------------------
        # print('kg_context size:',kg_context.size())
        cat_context = torch.cat([kg_context_pos, kg_context_neg], -1)
        # kg_context = self.ln_kg2(cat_context)
        # kg_context = self.dropout(kg_context)
        kg_context = F.leaky_relu(self.ln_kg2(cat_context))  # 120
        # kg_context = torch.dropout(kg_context, self.dropout, train=self.training)

        sub = txt_uniq - img_uniq_new
        #计算sub距离的绝对值
        # print('sub size',sub.size())#torch.Size([32,40])
        # sub_new = torch.sqrt(torch.sum(torch.pow(sub,2),dim=1))
        # print('sub_new:',sub_new)#torch.Size([32])
        # sub_new = sub_new.unsqueeze(1)

        hadmard = torch.mul(txt_share, img_share)
        # 计算点积距离的绝对值
        # print('hadmard size:',hadmard.size())#torch.Size([32,40])
        # hadmard_new = torch.sqrt(torch.sum(torch.pow(hadmard,2),dim=1))
        # print('hadmard_new:', hadmard_new)#torch.Size([32])
        # hadmard_new = hadmard_new.unsqueeze(1)
        # print('hadmard_new size:', hadmard_new.size())

        post_uniq_context = torch.cat([txt_uniq, sub, img_uniq_new], -1)  # 81
        post_share_context = torch.cat([txt_share, hadmard, img_share_new], -1)  # 81
        # print('model:',post_share_context.size(),post_uniq_context.size(),kg_context.size())
        cat = torch.cat([post_share_context, post_uniq_context, kg_context], -1)
        # print('cat size:', cat.size())
        output_class = self.rumor_classifier(cat)

        return output_class

#给图片输入加一个标识位，有图片增加标识位“1”，无图片增加标识位“0”；
class inconsistency_model_add_token(nn.Module):
    def __init__(self,config,pathset):
        #img_hidden_size, bert_path, hidden_dim, num_layers, dropout=0.5
        super(inconsistency_model_add_token, self).__init__()
        self.txt_hidden_dim = config.hidden_dim
        self.img_hidden_size = config.img_hidden_size
        self.bert_path = pathset.path_bert
        self.dropout = config.dropout
        self.num_layers = config.num_layers
        # self.clf = nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2,2)
        # self.clf = nn.Sequential(nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2+100, 1),
        #                          nn.Sigmoid())
        # self.dropout = dropout
        self.ln_txt = nn.Linear(self.txt_hidden_dim * 2, 200)
        self.ln_img = nn.Linear(self.img_hidden_size+1, 200)
        self.ln_shr = nn.Linear(200, 40, bias=False)
        self.ln_uniq_txt = nn.Linear(200, 40, bias=False)
        self.ln_uniq_img = nn.Linear(200, 40, bias=False)
        self.ln_kg1 = nn.Linear(50, 40)
        self.ln_kg2 = nn.Linear(160, 120)

        self.txtenc = Bert_lstm(self.txt_hidden_dim, self.bert_path, self.num_layers, self.dropout)
        self.imgenc = Resnet_Encoder()
        # 损失函数增加unique layer和shared layer
        # ---------------------------------
        # 多头attention
        self.num_head = 1
        self.dim_model = 80
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head
        self.fc_Q_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_Q_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.attention_pos = Scaled_Dot_Product_Attention_pos()
        self.attention_neg = Scaled_Dot_Product_Attention_neg()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.dim_model)
        #rumor_classifier
        # self.rumor_classifier = nn.Sequential(nn.Linear(360, 1),
        #                          nn.Sigmoid())
        self.rumor_classifier = nn.Sequential()
        self.rumor_classifier.add_module('r_fc1',nn.Linear(360,180))
        self.rumor_classifier.add_module('r_relu1',nn.LeakyReLU(True))
        self.rumor_classifier.add_module('r_fc2', nn.Linear(180, 1))
        self.rumor_classifier.add_module('r_softmax', nn.Sigmoid())

        #modal_classifier
        self.modal_classifier = nn.Sequential()
        self.modal_classifier.add_module('m_fc1', nn.Linear(40, 20))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.modal_classifier.add_module('m_relu1', nn.LeakyReLU(True))
        self.modal_classifier.add_module('m_fc2', nn.Linear(20, 1))
        self.modal_classifier.add_module('m_softmax', nn.Sigmoid())

    def forward(self, txt_token, txt_masks, img, kg1, kg2, kg_sim, y_img, args):
        # print('model:',kg1.size(),kg2.size(),kg_sim.size())
        #torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 1, 5])
        txt = self.txtenc(txt_token, txt_masks)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # print('txt.size:',txt.size())
        img = self.imgenc(img)
        # print('img size:',img.size())# torch.size([32,2048])
        # print('y_img:',y_img.size())#torch.size([32])
        #加入标识位 无图加‘0’，有图加‘1’
        y_img = y_img.unsqueeze(1)
        # print('y_img:',y_img.size())
        img_new = torch.cat([img,y_img],-1)
        # print('img_new size:',img_new.size())



        # img = torch.dropout(img, self.dropout, train=self.training)
        # print('img size:',img.size())
        # txt = self.ln_txt(txt)
        # txt = self.dropout(txt)
        txt = F.leaky_relu(self.ln_txt(txt))
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # txt = self.dropout(txt)
        # img = self.ln_img(img)
        # img = self.dropout(img)
        img = F.leaky_relu(self.ln_img(img_new))
        # img = torch.dropout(img, self.dropout, train=self.training)
        # img = self.dropout(img)
        txt_share = self.ln_shr(txt)
        # print('y_img size:',y_img.size()) #torch.Size([32])
        # print('txt_share size:',txt_share.size())#torch.Size([32,40])
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_share = self.ln_shr(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        txt_uniq = self.ln_uniq_txt(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_uniq = self.ln_uniq_img(img)
        # img = torch.dropout(img, self.dropout, train=self.training)

        modal_shr = torch.cat([txt_share, img_share], -1)  # 80
        # kg1 = self.ln_kg1(kg1)
        # kg1 = self.dropout(kg1)
        kg1 = F.leaky_relu(self.ln_kg1(kg1))
        # kg1 = torch.dropout(kg1, self.dropout, train=self.training)
        # kg2 = self.ln_kg1(kg2)
        # kg2 = self.dropout(kg2)
        kg2 = F.leaky_relu(self.ln_kg1(kg2))
        # kg2 = torch.dropout(kg2, self.dropout, train=self.training)
        # kg1 = self.ln_kg1(kg1)
        cat_kg = torch.cat([kg1, kg2], -1)  # 80
        # print('modal_shr size:',modal_shr.size())
        # print('cat_kg size:',cat_kg.size())
        # Q: modal_shr K:cat_kg V:cat_kg
        # ----------------------------------------------
        batch_size = cat_kg.size(0)
        Q_pos = self.fc_Q_pos(modal_shr)
        K_pos = self.fc_K_pos(cat_kg)
        V_pos = self.fc_V_pos(cat_kg)
        Q_pos = Q_pos.view(batch_size * self.num_head, -1, self.dim_head)
        K_pos = K_pos.view(batch_size * self.num_head, -1, self.dim_head)
        V_pos = V_pos.view(batch_size * self.num_head, -1, self.dim_head)

        Q_neg = self.fc_Q_neg(modal_shr)
        K_neg = self.fc_K_neg(cat_kg)
        V_neg = self.fc_V_neg(cat_kg)
        Q_neg = Q_neg.view(batch_size * self.num_head, -1, self.dim_head)
        K_neg = K_neg.view(batch_size * self.num_head, -1, self.dim_head)
        V_neg = V_neg.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K_pos.size(-1) ** -0.5  # 缩放因子
        kg_context_pos = self.attention_pos(Q_pos, K_pos, V_pos, scale, kg_sim)
        kg_context_pos = kg_context_pos.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_pos = self.fc1(kg_context_pos)
        kg_context_pos = self.dropout(kg_context_pos)
        # kg_context_pos = torch.dropout(kg_context_pos, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_pos = self.layer_norm(kg_context_pos)
        kg_context_pos = kg_context_pos.squeeze(1)

        kg_context_neg = self.attention_neg(Q_neg, K_neg, V_neg, scale, kg_sim)
        kg_context_neg = kg_context_neg.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_neg = self.fc2(kg_context_neg)
        kg_context_neg = self.dropout(kg_context_neg)
        # kg_cintext_neg = torch.dropout(kg_context_neg, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_neg = self.layer_norm(kg_context_neg)
        kg_context_neg = kg_context_neg.squeeze(1)
        # ------------------------------------------------------------------
        # print('kg_context size:',kg_context.size())
        cat_context = torch.cat([kg_context_pos, kg_context_neg], -1)
        # kg_context = self.ln_kg2(cat_context)
        # kg_context = self.dropout(kg_context)
        kg_context = F.leaky_relu(self.ln_kg2(cat_context))  # 120
        # kg_context = torch.dropout(kg_context, self.dropout, train=self.training)

        sub = txt_uniq - img_uniq
        #计算sub距离的绝对值
        # print('sub size',sub.size())#torch.Size([32,40])
        # sub_new = torch.sqrt(torch.sum(torch.pow(sub,2),dim=1))
        # print('sub_new:',sub_new)#torch.Size([32])
        # sub_new = sub_new.unsqueeze(1)

        hadmard = torch.mul(txt_share, img_share)
        # 计算点积距离的绝对值
        # print('hadmard size:',hadmard.size())#torch.Size([32,40])
        # hadmard_new = torch.sqrt(torch.sum(torch.pow(hadmard,2),dim=1))
        # print('hadmard_new:', hadmard_new)#torch.Size([32])
        # hadmard_new = hadmard_new.unsqueeze(1)
        # print('hadmard_new size:', hadmard_new.size())

        post_uniq_context = torch.cat([txt_uniq, sub, img_uniq], -1)  # 81
        post_share_context = torch.cat([txt_share, hadmard, img_share], -1)  # 81
        # print('model:',post_share_context.size(),post_uniq_context.size(),kg_context.size())
        cat = torch.cat([post_share_context, post_uniq_context, kg_context], -1)
        # print('cat size:', cat.size())
        output_class = self.rumor_classifier(cat)

        return output_class

class inconsistency_model_add_token_overfitting(nn.Module):
    def __init__(self,config,pathset):
        #img_hidden_size, bert_path, hidden_dim, num_layers, dropout=0.5
        super(inconsistency_model_add_token_overfitting, self).__init__()
        self.txt_hidden_dim = config.hidden_dim
        self.img_hidden_size = config.img_hidden_size
        self.bert_path = pathset.path_bert
        self.dropout = config.dropout
        self.num_layers = config.num_layers
        # self.clf = nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2,2)
        # self.clf = nn.Sequential(nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2+100, 1),
        #                          nn.Sigmoid())
        # self.dropout = dropout
        self.ln_txt = nn.Linear(self.txt_hidden_dim * 2, 200)
        self.ln_img = nn.Linear(self.img_hidden_size+1, 200)
        self.ln_shr = nn.Linear(200, 40, bias=False)
        self.ln_uniq_txt = nn.Linear(200, 40, bias=False)
        self.ln_uniq_img = nn.Linear(200, 40, bias=False)
        self.ln_kg1 = nn.Linear(50, 40)
        self.ln_kg2 = nn.Linear(160, 120)

        self.txtenc = Bert_lstm(self.txt_hidden_dim, self.bert_path, self.num_layers, self.dropout)
        self.imgenc = Resnet_Encoder()
        # 损失函数增加unique layer和shared layer
        # ---------------------------------
        # 多头attention
        self.num_head = 1
        self.dim_model = 80
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head
        self.fc_Q_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_Q_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.attention_pos = Scaled_Dot_Product_Attention_pos()
        self.attention_neg = Scaled_Dot_Product_Attention_neg()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.dim_model)
        #rumor_classifier
        # self.rumor_classifier = nn.Sequential(nn.Linear(360, 1),
        #                          nn.Sigmoid())
        self.rumor_classifier = nn.Sequential()
        self.rumor_classifier.add_module('r_fc1',nn.Linear(360,180))
        self.rumor_classifier.add_module('r_relu1',nn.LeakyReLU(True))
        self.rumor_classifier.add_module('r_fc2', nn.Linear(180, 1))
        self.rumor_classifier.add_module('r_softmax', nn.Sigmoid())

        #modal_classifier
        self.modal_classifier = nn.Sequential()
        self.modal_classifier.add_module('m_fc1', nn.Linear(40, 20))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.modal_classifier.add_module('m_relu1', nn.LeakyReLU(True))
        self.modal_classifier.add_module('m_fc2', nn.Linear(20, 1))
        self.modal_classifier.add_module('m_softmax', nn.Sigmoid())

    def forward(self, txt_token, txt_masks, img, kg1, kg2, kg_sim, y_img, args):
        # print('model:',kg1.size(),kg2.size(),kg_sim.size())
        #torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 1, 5])
        txt = self.txtenc(txt_token, txt_masks)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # print('txt.size:',txt.size())
        img = self.imgenc(img)
        # print('img size:',img.size())# torch.size([32,2048])
        # print('y_img:',y_img.size())#torch.size([32])
        #加入标识位 无图加‘0’，有图加‘1’
        y_img = y_img.unsqueeze(1)
        # print('y_img:',y_img.size())
        img_new = torch.cat([img,y_img],-1)
        # print('img_new size:',img_new.size())



        # img = torch.dropout(img, self.dropout, train=self.training)
        # print('img size:',img.size())
        txt = self.ln_txt(txt)
        txt = self.dropout(txt)
        txt = F.leaky_relu(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # txt = self.dropout(txt)
        img_linear = self.ln_img(img_new)
        img_linear = self.dropout(img_linear)
        img = F.leaky_relu(img_linear)
        # img = torch.dropout(img, self.dropout, train=self.training)
        # img = self.dropout(img)
        txt_share = self.ln_shr(txt)
        # print('y_img size:',y_img.size()) #torch.Size([32])
        # print('txt_share size:',txt_share.size())#torch.Size([32,40])
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        txt_share = self.dropout(txt_share)

        img_share = self.ln_shr(img)
        img_share = self.dropout(img_share)
        # img = torch.dropout(img, self.dropout, train=self.training)
        txt_uniq = self.ln_uniq_txt(txt)
        txt_uniq = self.dropout(txt_uniq)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_uniq = self.ln_uniq_img(img)
        img_uniq = self.dropout(img_uniq)
        # img = torch.dropout(img, self.dropout, train=self.training)

        modal_shr = torch.cat([txt_share, img_share], -1)  # 80
        kg1 = self.ln_kg1(kg1)
        kg1 = self.dropout(kg1)
        kg1 = F.leaky_relu(kg1)
        # kg1 = torch.dropout(kg1, self.dropout, train=self.training)
        kg2 = self.ln_kg1(kg2)
        kg2 = self.dropout(kg2)
        kg2 = F.leaky_relu(kg2)
        # kg2 = torch.dropout(kg2, self.dropout, train=self.training)
        # kg1 = self.ln_kg1(kg1)
        cat_kg = torch.cat([kg1, kg2], -1)  # 80
        # print('modal_shr size:',modal_shr.size())
        # print('cat_kg size:',cat_kg.size())
        # Q: modal_shr K:cat_kg V:cat_kg
        # ----------------------------------------------
        batch_size = cat_kg.size(0)
        Q_pos = self.fc_Q_pos(modal_shr)
        K_pos = self.fc_K_pos(cat_kg)
        V_pos = self.fc_V_pos(cat_kg)
        Q_pos = Q_pos.view(batch_size * self.num_head, -1, self.dim_head)
        K_pos = K_pos.view(batch_size * self.num_head, -1, self.dim_head)
        V_pos = V_pos.view(batch_size * self.num_head, -1, self.dim_head)

        Q_neg = self.fc_Q_neg(modal_shr)
        K_neg = self.fc_K_neg(cat_kg)
        V_neg = self.fc_V_neg(cat_kg)
        Q_neg = Q_neg.view(batch_size * self.num_head, -1, self.dim_head)
        K_neg = K_neg.view(batch_size * self.num_head, -1, self.dim_head)
        V_neg = V_neg.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K_pos.size(-1) ** -0.5  # 缩放因子
        kg_context_pos = self.attention_pos(Q_pos, K_pos, V_pos, scale, kg_sim)
        kg_context_pos = kg_context_pos.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_pos = self.fc1(kg_context_pos)
        kg_context_pos = self.dropout(kg_context_pos)
        # kg_context_pos = torch.dropout(kg_context_pos, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_pos = self.layer_norm(kg_context_pos)
        kg_context_pos = kg_context_pos.squeeze(1)

        kg_context_neg = self.attention_neg(Q_neg, K_neg, V_neg, scale, kg_sim)
        kg_context_neg = kg_context_neg.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_neg = self.fc2(kg_context_neg)
        kg_context_neg = self.dropout(kg_context_neg)
        # kg_cintext_neg = torch.dropout(kg_context_neg, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_neg = self.layer_norm(kg_context_neg)
        kg_context_neg = kg_context_neg.squeeze(1)
        # ------------------------------------------------------------------
        # print('kg_context size:',kg_context.size())
        cat_context = torch.cat([kg_context_pos, kg_context_neg], -1)
        kg_context = self.ln_kg2(cat_context)
        kg_context = self.dropout(kg_context)
        kg_context = F.leaky_relu(kg_context)  # 120
        # kg_context = torch.dropout(kg_context, self.dropout, train=self.training)

        sub = txt_uniq - img_uniq
        #计算sub距离的绝对值
        # print('sub size',sub.size())#torch.Size([32,40])
        # sub_new = torch.sqrt(torch.sum(torch.pow(sub,2),dim=1))
        # print('sub_new:',sub_new)#torch.Size([32])
        # sub_new = sub_new.unsqueeze(1)

        hadmard = torch.mul(txt_share, img_share)
        # 计算点积距离的绝对值
        # print('hadmard size:',hadmard.size())#torch.Size([32,40])
        # hadmard_new = torch.sqrt(torch.sum(torch.pow(hadmard,2),dim=1))
        # print('hadmard_new:', hadmard_new)#torch.Size([32])
        # hadmard_new = hadmard_new.unsqueeze(1)
        # print('hadmard_new size:', hadmard_new.size())

        post_uniq_context = torch.cat([txt_uniq, sub, img_uniq], -1)  # 81
        post_share_context = torch.cat([txt_share, hadmard, img_share], -1)  # 81
        # print('model:',post_share_context.size(),post_uniq_context.size(),kg_context.size())
        cat = torch.cat([post_share_context, post_uniq_context, kg_context], -1)
        # print('cat size:', cat.size())
        output_class = self.rumor_classifier(cat)

        return output_class

#emnlp:kdin初始模型
class inconsistency_model_kdin(nn.Module):
    def __init__(self,config,pathset):
        #img_hidden_size, bert_path, hidden_dim, num_layers, dropout=0.5
        super(inconsistency_model_kdin, self).__init__()
        self.txt_hidden_dim = config.hidden_dim
        self.img_hidden_size = config.img_hidden_size
        self.bert_path = pathset.path_bert
        self.dropout = config.dropout
        self.num_layers = config.num_layers
        # self.clf = nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2,2)
        # self.clf = nn.Sequential(nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2+100, 1),
        #                          nn.Sigmoid())
        # self.dropout = dropout
        self.ln_txt = nn.Linear(self.txt_hidden_dim * 2, 200)
        self.ln_img = nn.Linear(self.img_hidden_size, 200)
        self.ln_shr = nn.Linear(200, 40, bias=False)
        self.ln_uniq_txt = nn.Linear(200, 40, bias=False)
        self.ln_uniq_img = nn.Linear(200, 40, bias=False)
        self.ln_kg1 = nn.Linear(50, 40)
        self.ln_kg2 = nn.Linear(160, 120)

        self.txtenc = Bert_lstm(self.txt_hidden_dim, self.bert_path, self.num_layers, self.dropout)
        self.imgenc = Resnet_Encoder()
        # 损失函数增加unique layer和shared layer
        # ---------------------------------
        # 多头attention
        self.num_head = 1
        self.dim_model = 80
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head
        self.fc_Q_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_Q_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.attention_pos = Scaled_Dot_Product_Attention_pos()
        self.attention_neg = Scaled_Dot_Product_Attention_neg()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.dim_model)
        #rumor_classifier
        # self.rumor_classifier = nn.Sequential(nn.Linear(360, 1),
        #                          nn.Sigmoid())
        self.rumor_classifier = nn.Sequential()
        self.rumor_classifier.add_module('r_fc1',nn.Linear(360,180))
        self.rumor_classifier.add_module('r_relu1',nn.LeakyReLU(True))
        self.rumor_classifier.add_module('r_fc2', nn.Linear(180, 1))
        self.rumor_classifier.add_module('r_softmax', nn.Sigmoid())

        #modal_classifier
        self.modal_classifier = nn.Sequential()
        self.modal_classifier.add_module('m_fc1', nn.Linear(40, 20))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.modal_classifier.add_module('m_relu1', nn.LeakyReLU(True))
        self.modal_classifier.add_module('m_fc2', nn.Linear(20, 1))
        self.modal_classifier.add_module('m_softmax', nn.Sigmoid())

    def forward(self, txt_token, txt_masks, img, kg1, kg2, kg_sim, y_img, args):
        # print('model:',kg1.size(),kg2.size(),kg_sim.size())
        #torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 1, 5])
        txt = self.txtenc(txt_token, txt_masks)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # print('txt.size:',txt.size())
        img = self.imgenc(img)
        # print('img size:',img.size())# torch.size([32,2048])
        # print('y_img:',y_img.size())#torch.size([32])
        #加入标识位 无图加‘0’，有图加‘1’
        # y_img = y_img.unsqueeze(1)
        # print('y_img:',y_img.size())
        # img_new = torch.cat([img,y_img],-1)
        # print('img_new size:',img_new.size())
        img_new = img



        # img = torch.dropout(img, self.dropout, train=self.training)
        # print('img size:',img.size())
        # txt = self.ln_txt(txt)
        # txt = self.dropout(txt)
        txt = F.leaky_relu(self.ln_txt(txt))
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # txt = self.dropout(txt)
        # img = self.ln_img(img)
        # img = self.dropout(img)
        img = F.leaky_relu(self.ln_img(img_new))
        # img = torch.dropout(img, self.dropout, train=self.training)
        # img = self.dropout(img)
        txt_share = self.ln_shr(txt)
        # print('y_img size:',y_img.size()) #torch.Size([32])
        # print('txt_share size:',txt_share.size())#torch.Size([32,40])
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_share = self.ln_shr(img)
        # img = torch.dropout(img, self.dropout, train=self.training)
        txt_uniq = self.ln_uniq_txt(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_uniq = self.ln_uniq_img(img)
        # img = torch.dropout(img, self.dropout, train=self.training)

        modal_shr = torch.cat([txt_share, img_share], -1)  # 80
        # kg1 = self.ln_kg1(kg1)
        # kg1 = self.dropout(kg1)
        kg1 = F.leaky_relu(self.ln_kg1(kg1))
        # kg1 = torch.dropout(kg1, self.dropout, train=self.training)
        # kg2 = self.ln_kg1(kg2)
        # kg2 = self.dropout(kg2)
        kg2 = F.leaky_relu(self.ln_kg1(kg2))
        # kg2 = torch.dropout(kg2, self.dropout, train=self.training)
        # kg1 = self.ln_kg1(kg1)
        cat_kg = torch.cat([kg1, kg2], -1)  # 80
        # print('modal_shr size:',modal_shr.size())
        # print('cat_kg size:',cat_kg.size())
        # Q: modal_shr K:cat_kg V:cat_kg
        # ----------------------------------------------
        batch_size = cat_kg.size(0)
        Q_pos = self.fc_Q_pos(modal_shr)
        K_pos = self.fc_K_pos(cat_kg)
        V_pos = self.fc_V_pos(cat_kg)
        Q_pos = Q_pos.view(batch_size * self.num_head, -1, self.dim_head)
        K_pos = K_pos.view(batch_size * self.num_head, -1, self.dim_head)
        V_pos = V_pos.view(batch_size * self.num_head, -1, self.dim_head)

        Q_neg = self.fc_Q_neg(modal_shr)
        K_neg = self.fc_K_neg(cat_kg)
        V_neg = self.fc_V_neg(cat_kg)
        Q_neg = Q_neg.view(batch_size * self.num_head, -1, self.dim_head)
        K_neg = K_neg.view(batch_size * self.num_head, -1, self.dim_head)
        V_neg = V_neg.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K_pos.size(-1) ** -0.5  # 缩放因子
        kg_context_pos = self.attention_pos(Q_pos, K_pos, V_pos, scale, kg_sim)
        kg_context_pos = kg_context_pos.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_pos = self.fc1(kg_context_pos)
        kg_context_pos = self.dropout(kg_context_pos)
        # kg_context_pos = torch.dropout(kg_context_pos, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_pos = self.layer_norm(kg_context_pos)
        kg_context_pos = kg_context_pos.squeeze(1)

        kg_context_neg = self.attention_neg(Q_neg, K_neg, V_neg, scale, kg_sim)
        kg_context_neg = kg_context_neg.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_neg = self.fc2(kg_context_neg)
        kg_context_neg = self.dropout(kg_context_neg)
        # kg_cintext_neg = torch.dropout(kg_context_neg, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_neg = self.layer_norm(kg_context_neg)
        kg_context_neg = kg_context_neg.squeeze(1)
        # ------------------------------------------------------------------
        # print('kg_context size:',kg_context.size())
        cat_context = torch.cat([kg_context_pos, kg_context_neg], -1)
        # kg_context = self.ln_kg2(cat_context)
        # kg_context = self.dropout(kg_context)
        kg_context = F.leaky_relu(self.ln_kg2(cat_context))  # 120
        # kg_context = torch.dropout(kg_context, self.dropout, train=self.training)

        sub = txt_uniq - img_uniq
        #计算sub距离的绝对值
        # print('sub size',sub.size())#torch.Size([32,40])
        # sub_new = torch.sqrt(torch.sum(torch.pow(sub,2),dim=1))
        # print('sub_new:',sub_new)#torch.Size([32])
        # sub_new = sub_new.unsqueeze(1)

        hadmard = torch.mul(txt_share, img_share)
        # 计算点积距离的绝对值
        # print('hadmard size:',hadmard.size())#torch.Size([32,40])
        # hadmard_new = torch.sqrt(torch.sum(torch.pow(hadmard,2),dim=1))
        # print('hadmard_new:', hadmard_new)#torch.Size([32])
        # hadmard_new = hadmard_new.unsqueeze(1)
        # print('hadmard_new size:', hadmard_new.size())

        post_uniq_context = torch.cat([txt_uniq, sub, img_uniq], -1)  # 81
        post_share_context = torch.cat([txt_share, hadmard, img_share], -1)  # 81
        # print('model:',post_share_context.size(),post_uniq_context.size(),kg_context.size())
        cat = torch.cat([post_share_context, post_uniq_context, kg_context], -1)
        # print('cat size:', cat.size())
        output_class = self.rumor_classifier(cat)

        return output_class

class inconsistency_model_kdin_overfitting(nn.Module):
    def __init__(self,config,pathset):
        #img_hidden_size, bert_path, hidden_dim, num_layers, dropout=0.5
        super(inconsistency_model_kdin_overfitting, self).__init__()
        self.txt_hidden_dim = config.hidden_dim
        self.img_hidden_size = config.img_hidden_size
        self.bert_path = pathset.path_bert
        self.dropout = config.dropout
        self.num_layers = config.num_layers
        # self.clf = nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2,2)
        # self.clf = nn.Sequential(nn.Linear(self.img_hidden_size+self.txt_hidden_dim*2+100, 1),
        #                          nn.Sigmoid())
        # self.dropout = dropout
        self.ln_txt = nn.Linear(self.txt_hidden_dim * 2, 200)
        self.ln_img = nn.Linear(self.img_hidden_size, 200)
        self.ln_shr = nn.Linear(200, 40, bias=False)
        self.ln_uniq_txt = nn.Linear(200, 40, bias=False)
        self.ln_uniq_img = nn.Linear(200, 40, bias=False)
        self.ln_kg1 = nn.Linear(50, 40)
        self.ln_kg2 = nn.Linear(160, 120)

        self.txtenc = Bert_lstm(self.txt_hidden_dim, self.bert_path, self.num_layers, self.dropout)
        self.imgenc = Resnet_Encoder()
        # 损失函数增加unique layer和shared layer
        # ---------------------------------
        # 多头attention
        self.num_head = 1
        self.dim_model = 80
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head
        self.fc_Q_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_pos = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_Q_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V_neg = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.attention_pos = Scaled_Dot_Product_Attention_pos()
        self.attention_neg = Scaled_Dot_Product_Attention_neg()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.dim_model)
        #rumor_classifier
        # self.rumor_classifier = nn.Sequential(nn.Linear(360, 1),
        #                          nn.Sigmoid())
        self.rumor_classifier = nn.Sequential()
        self.rumor_classifier.add_module('r_fc1',nn.Linear(360,180))
        self.rumor_classifier.add_module('r_relu1',nn.LeakyReLU(True))
        self.rumor_classifier.add_module('r_fc2', nn.Linear(180, 1))
        self.rumor_classifier.add_module('r_softmax', nn.Sigmoid())

        #modal_classifier
        self.modal_classifier = nn.Sequential()
        self.modal_classifier.add_module('m_fc1', nn.Linear(40, 20))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.modal_classifier.add_module('m_relu1', nn.LeakyReLU(True))
        self.modal_classifier.add_module('m_fc2', nn.Linear(20, 1))
        self.modal_classifier.add_module('m_softmax', nn.Sigmoid())

    def forward(self, txt_token, txt_masks, img, kg1, kg2, kg_sim, y_img, args):
        # print('model:',kg1.size(),kg2.size(),kg_sim.size())
        #torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 5, 50]) torch.Size([16, 1, 1, 5])
        txt = self.txtenc(txt_token, txt_masks)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # print('txt.size:',txt.size())
        img = self.imgenc(img)
        # print('img size:',img.size())# torch.size([32,2048])
        # print('y_img:',y_img.size())#torch.size([32])
        #加入标识位 无图加‘0’，有图加‘1’
        # y_img = y_img.unsqueeze(1)
        # print('y_img:',y_img.size())
        # img_new = torch.cat([img,y_img],-1)
        # print('img_new size:',img_new.size())
        img_new = img



        # img = torch.dropout(img, self.dropout, train=self.training)
        # print('img size:',img.size())
        txt = self.ln_txt(txt)
        txt = self.dropout(txt)
        txt = F.leaky_relu(txt)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        # txt = self.dropout(txt)
        img_linear = self.ln_img(img_new)
        img_linear = self.dropout(img_linear)
        img = F.leaky_relu(img_linear)
        # img = torch.dropout(img, self.dropout, train=self.training)
        # img = self.dropout(img)
        txt_share = self.ln_shr(txt)
        # print('y_img size:',y_img.size()) #torch.Size([32])
        # print('txt_share size:',txt_share.size())#torch.Size([32,40])
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        txt_share = self.dropout(txt_share)

        img_share = self.ln_shr(img)
        img_share = self.dropout(img_share)
        # img = torch.dropout(img, self.dropout, train=self.training)
        txt_uniq = self.ln_uniq_txt(txt)
        txt_uniq = self.dropout(txt_uniq)
        # txt = torch.dropout(txt, self.dropout, train=self.training)
        img_uniq = self.ln_uniq_img(img)
        img_uniq = self.dropout(img_uniq)
        # img = torch.dropout(img, self.dropout, train=self.training)

        modal_shr = torch.cat([txt_share, img_share], -1)  # 80
        kg1 = self.ln_kg1(kg1)
        kg1 = self.dropout(kg1)
        kg1 = F.leaky_relu(kg1)
        # kg1 = torch.dropout(kg1, self.dropout, train=self.training)
        kg2 = self.ln_kg1(kg2)
        kg2 = self.dropout(kg2)
        kg2 = F.leaky_relu(kg2)
        # kg2 = torch.dropout(kg2, self.dropout, train=self.training)
        # kg1 = self.ln_kg1(kg1)
        cat_kg = torch.cat([kg1, kg2], -1)  # 80
        # print('modal_shr size:',modal_shr.size())
        # print('cat_kg size:',cat_kg.size())
        # Q: modal_shr K:cat_kg V:cat_kg
        # ----------------------------------------------
        batch_size = cat_kg.size(0)
        Q_pos = self.fc_Q_pos(modal_shr)
        K_pos = self.fc_K_pos(cat_kg)
        V_pos = self.fc_V_pos(cat_kg)
        Q_pos = Q_pos.view(batch_size * self.num_head, -1, self.dim_head)
        K_pos = K_pos.view(batch_size * self.num_head, -1, self.dim_head)
        V_pos = V_pos.view(batch_size * self.num_head, -1, self.dim_head)

        Q_neg = self.fc_Q_neg(modal_shr)
        K_neg = self.fc_K_neg(cat_kg)
        V_neg = self.fc_V_neg(cat_kg)
        Q_neg = Q_neg.view(batch_size * self.num_head, -1, self.dim_head)
        K_neg = K_neg.view(batch_size * self.num_head, -1, self.dim_head)
        V_neg = V_neg.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K_pos.size(-1) ** -0.5  # 缩放因子
        kg_context_pos = self.attention_pos(Q_pos, K_pos, V_pos, scale, kg_sim)
        kg_context_pos = kg_context_pos.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_pos = self.fc1(kg_context_pos)
        kg_context_pos = self.dropout(kg_context_pos)
        # kg_context_pos = torch.dropout(kg_context_pos, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_pos = self.layer_norm(kg_context_pos)
        kg_context_pos = kg_context_pos.squeeze(1)

        kg_context_neg = self.attention_neg(Q_neg, K_neg, V_neg, scale, kg_sim)
        kg_context_neg = kg_context_neg.view(batch_size, -1, self.dim_head * self.num_head)
        kg_context_neg = self.fc2(kg_context_neg)
        kg_context_neg = self.dropout(kg_context_neg)
        # kg_cintext_neg = torch.dropout(kg_context_neg, self.dropout, train=self.training)
        # out = out + x  # 残差连接
        kg_context_neg = self.layer_norm(kg_context_neg)
        kg_context_neg = kg_context_neg.squeeze(1)
        # ------------------------------------------------------------------
        # print('kg_context size:',kg_context.size())
        cat_context = torch.cat([kg_context_pos, kg_context_neg], -1)
        kg_context = self.ln_kg2(cat_context)
        kg_context = self.dropout(kg_context)
        kg_context = F.leaky_relu(kg_context)  # 120
        # kg_context = torch.dropout(kg_context, self.dropout, train=self.training)

        sub = txt_uniq - img_uniq
        #计算sub距离的绝对值
        # print('sub size',sub.size())#torch.Size([32,40])
        # sub_new = torch.sqrt(torch.sum(torch.pow(sub,2),dim=1))
        # print('sub_new:',sub_new)#torch.Size([32])
        # sub_new = sub_new.unsqueeze(1)

        hadmard = torch.mul(txt_share, img_share)
        # 计算点积距离的绝对值
        # print('hadmard size:',hadmard.size())#torch.Size([32,40])
        # hadmard_new = torch.sqrt(torch.sum(torch.pow(hadmard,2),dim=1))
        # print('hadmard_new:', hadmard_new)#torch.Size([32])
        # hadmard_new = hadmard_new.unsqueeze(1)
        # print('hadmard_new size:', hadmard_new.size())

        post_uniq_context = torch.cat([txt_uniq, sub, img_uniq], -1)  # 81
        post_share_context = torch.cat([txt_share, hadmard, img_share], -1)  # 81
        # print('model:',post_share_context.size(),post_uniq_context.size(),kg_context.size())
        cat = torch.cat([post_share_context, post_uniq_context, kg_context], -1)
        # print('cat size:', cat.size())
        output_class = self.rumor_classifier(cat)

        return output_class