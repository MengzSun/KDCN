import torch
from gensim.models import Word2Vec
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#随机
import random
from random import shuffle
import csv


import torchtext.vocab as vocab
from random_5fold import *
import warnings
warnings.filterwarnings('ignore')

#weibo\pheme
class data_preprocess_ATT_bert_nfold():
    def __init__(self,sentences,images,mids,y,y_img,config,pathset): #sentence==token_ids
        #glove_path = '/home/sunmengzhu2019/kg_rumor/w2v/merge_sgns_bigram_char300.txt',
        #transE_path = '/home/sunmengzhu2019/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        # self.glove_path = pathset.path_glove
        self.sentences = sentences
        self.images = images
        self.mids = mids
        self.y = y
        self.y_img = y_img

        self.sen_len = config.sen_len
        self.entity_len = config.entity_len

        self.transE_path = pathset.path_transe
        self.pathset = pathset
        self.config = config

        self.index2mid = []
        self.mid2index = {}
        self.index2word = []
        self.word2index = {}
        # self.embedding_matrix = []
        # self.embedding_glove = []

    def get_transe(self):
        vec = np.memmap(self.transE_path, dtype='float32', mode='r')
        # row = int(vec.shape[0]/50)
        self.embedding = vec.reshape((-1,50))
        self.embedding_entity_dim = 50 #50

        self.add_entity_embedding("<pad1>")
        self.add_entity_embedding("<pad2>")
        self.add_entity_embedding("<pad3>")
        self.add_entity_embedding("<pad4>")
        # #归一化
        # min_max_scaler = MinMaxScaler()
        # self.embedding_minmax = min_max_scaler.fit_transform(self.embedding)
        #标准化
        # mu = np.mean(self.embedding, axis=0)
        # std = np.std(self.embedding, axis=0)
        # self.embedding_mnstd = (self.embedding - mu) / std
        # self.embedding_fin = torch.from_numpy(self.embedding_mnstd)
        # return torch.tensor(self.embedding_fin)
        return torch.tensor(self.embedding)

    def make_dic(self):
        dic_path = self.pathset.path_dic
        with open(dic_path,'r',encoding='utf-8')as t:
            text_all = t.readlines()
        for line in text_all[1:]:
            mid = line.split('\t')[0]
            idx = line.split('\t')[1].strip('\n')
            self.mid2index[mid] = int(idx)
        return self.mid2index

#TODO:random的问题
    def add_entity_embedding(self,entity):
        vec_entity = torch.empty(1, 50)
        nn.init.uniform_(vec_entity)
        self.mid2index[entity] = len(self.mid2index)
        # self.index2mid.append(entity)
        vec_entity = vec_entity.numpy()
        self.embedding = np.concatenate([self.embedding, vec_entity], 0)

    def top_dis_entity_cos(self,mid_post_index,embedding):
        # top_dis = 0
        top_dis = -1
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() > top_dis:
                        top_dis = cos_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def last_dis_entity_cos(self,mid_post_index,embedding):
        last_dis = 1
        last_vec1 = 0
        last_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() < last_dis:
                        last_dis = cos_sim.item()
                        last_vec1 = int(id1)
                        last_vec2 = int(id2)
                else:
                    continue
        print(last_dis)
        mid_post_list = []
        mid_post_list.append(last_vec1)
        mid_post_list.append(last_vec2)
        return mid_post_list

    def top_dis_entity_manhattan(self,mid_post_index, embedding):
        top_dis = 0
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    vec_sub = vec1-vec2
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    if manhattan_sim.item() > top_dis:
                        top_dis = manhattan_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def top_dis_5entity_cos(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    top_dis.append(cos_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[cos_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list

    def top_dis_5entity_manhattan(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    vec_sub = vec1 - vec2
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    top_dis.append(manhattan_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[manhattan_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        sim_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])
                sim_post_list.append(sim)

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list, sim_post_list

    def pad_eneity(self,mid,mid2index):
        if len(mid) < self.entity_len:
            pad_len = self.entity_len-len(mid)
            for j in range(pad_len):
                mid.append(mid2index["<pad{}>".format(j+1)])
        # assert len(mid) == self.entity_len
        # print(mid)
        return mid

    def img_trans(self):
        #定义一个变量transform，是对输入图像进行的所有图像转换的组合
        transform = transforms.Compose([
            transforms.Resize(256),
            #图像调整为256x256像素
            transforms.CenterCrop(224),
            #将图像中心裁剪出来，大小为224x224像素
            transforms.ToTensor(),
            #图像转换为tensor数据类型
            #将图像的平均值和标准差设置为指定的值来正则化图像
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )])
        return transform

    def load_data(self):
        #对bert处理
        # segments = []
        attention_masks = []
        input_ids = []
        tokens = self.sentences
        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in tokens:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids.append(tkn['input_ids'].squeeze(0))
            attention_masks.append(tkn['attention_mask'].squeeze(0))

        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        #对label处理
        y = [int(label) for label in self.y]
        y = torch.LongTensor(y)

        #对img_label处理
        y_img = [int(label) for label in self.y_img]
        y_img = torch.LongTensor(y_img)

        #对实体mids进行处理
        mid2index = self.make_dic()
        embedding = self.get_transe()  # torch.Size([86054151, 50])
        # print('transe size:',embedding.size())
        # count = 0
        vec_all_1, vec_all_2, sim_all = [],[],[]

        for i, mid_post in enumerate(self.mids):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1.append(vec_post_1)
            vec_all_2.append(vec_post_2)
            sim_all.append(sim_post)
            # print(i)
        # return torch.LongTensor(vec_all)
        # print("vec_all_1",vec_all_1)
        # print("vec_all_2",vec_all_2)

        #train\val\test
        fold0_test, fold0_val, fold0_train, \
        fold1_test, fold1_val, fold1_train, \
        fold2_test, fold2_val, fold2_train, \
        fold3_test, fold3_val, fold3_train, \
        fold4_test, fold4_val, fold4_train = split_data_5fold(len(y), y,self.config.train,self.config.val,\
                                                              self.config.test, shuffle=True)

        train_dict_0, val_dict_0, test_dict_0 = {}, {}, {}
        train_dict_1, val_dict_1, test_dict_1 = {}, {}, {}
        train_dict_2, val_dict_2, test_dict_2 = {}, {}, {}
        train_dict_3, val_dict_3, test_dict_3 = {}, {}, {}
        train_dict_4, val_dict_4, test_dict_4 = {}, {}, {}

        names_dict = {'input_ids':input_ids,'attention_masks':attention_masks,'image':self.images,\
                      'vec_1':vec_all_1, 'vec_2':vec_all_2, 'sim_list':sim_all,'y':y,'y_img':y_img}
        for name in names_dict:
            train_dict_0[name] = [names_dict[name][i] for i in fold0_train]
            val_dict_0[name] = [names_dict[name][i] for i in fold0_val]
            test_dict_0[name] = [names_dict[name][i] for i in fold0_test]
        for name in names_dict:
            train_dict_1[name] = [names_dict[name][i] for i in fold1_train]
            val_dict_1[name] = [names_dict[name][i] for i in fold1_val]
            test_dict_1[name] = [names_dict[name][i] for i in fold1_test]
        for name in names_dict:
            train_dict_2[name] = [names_dict[name][i] for i in fold2_train]
            val_dict_2[name] = [names_dict[name][i] for i in fold2_val]
            test_dict_2[name] = [names_dict[name][i] for i in fold2_test]
        for name in names_dict:
            train_dict_3[name] = [names_dict[name][i] for i in fold3_train]
            val_dict_3[name] = [names_dict[name][i] for i in fold3_val]
            test_dict_3[name] = [names_dict[name][i] for i in fold3_test]
        for name in names_dict:
            train_dict_4[name] = [names_dict[name][i] for i in fold4_train]
            val_dict_4[name] = [names_dict[name][i] for i in fold4_val]
            test_dict_4[name] = [names_dict[name][i] for i in fold4_test]


        return train_dict_0, val_dict_0, test_dict_0, \
               train_dict_1, val_dict_1, test_dict_1, \
               train_dict_2, val_dict_2, test_dict_2, \
               train_dict_3, val_dict_3, test_dict_3, \
               train_dict_4, val_dict_4, test_dict_4
#twitter
class data_preprocess_ATT_bert():
    def __init__(self,sentences_train,images_train,mids_train,y_train,y_img_train,\
                 sentences_test,images_test,mids_test,y_test,y_img_test,config,pathset): #sentence==token_ids
        #glove_path = '/home/sunmengzhu2019/kg_rumor/w2v/merge_sgns_bigram_char300.txt',
        #transE_path = '/home/sunmengzhu2019/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        # self.glove_path = pathset.path_glove
        self.sentences_train = sentences_train
        self.images_train = images_train
        self.mids_train = mids_train
        self.y_train = y_train
        self.y_img_train = y_img_train

        self.sentences_test = sentences_test
        self.images_test = images_test
        self.mids_test = mids_test
        self.y_test = y_test
        self.y_img_test = y_img_test

        self.sen_len = config.sen_len
        self.entity_len = config.entity_len

        self.transE_path = pathset.path_transe
        self.pathset = pathset
        self.config = config

        self.index2mid = []
        self.mid2index = {}
        self.index2word = []
        self.word2index = {}
        # self.embedding_matrix = []
        # self.embedding_glove = []

    def get_transe(self):
        vec = np.memmap(self.transE_path, dtype='float32', mode='r')
        # row = int(vec.shape[0]/50)
        self.embedding = vec.reshape((-1,50))
        self.embedding_entity_dim = 50 #50

        self.add_entity_embedding("<pad1>")
        self.add_entity_embedding("<pad2>")
        self.add_entity_embedding("<pad3>")
        self.add_entity_embedding("<pad4>")
        # #归一化
        # min_max_scaler = MinMaxScaler()
        # self.embedding_minmax = min_max_scaler.fit_transform(self.embedding)
        #标准化
        # mu = np.mean(self.embedding, axis=0)
        # std = np.std(self.embedding, axis=0)
        # self.embedding_mnstd = (self.embedding - mu) / std
        # self.embedding_fin = torch.from_numpy(self.embedding_mnstd)
        # return torch.tensor(self.embedding_fin)
        return torch.tensor(self.embedding)

    def make_dic(self):
        dic_path = self.pathset.path_dic
        with open(dic_path,'r',encoding='utf-8')as t:
            text_all = t.readlines()
        for line in text_all[1:]:
            mid = line.split('\t')[0]
            idx = line.split('\t')[1].strip('\n')
            self.mid2index[mid] = int(idx)
        return self.mid2index

#TODO:RANDOM的问题
    def add_entity_embedding(self,entity):
        vec_entity = torch.empty(1, 50)
        nn.init.uniform_(vec_entity)
        self.mid2index[entity] = len(self.mid2index)
        # self.index2mid.append(entity)
        vec_entity = vec_entity.numpy()
        self.embedding = np.concatenate([self.embedding, vec_entity], 0)

    def top_dis_entity_cos(self,mid_post_index,embedding):
        # top_dis = 0
        top_dis = -1
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() > top_dis:
                        top_dis = cos_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def last_dis_entity_cos(self,mid_post_index,embedding):
        last_dis = 1
        last_vec1 = 0
        last_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() < last_dis:
                        last_dis = cos_sim.item()
                        last_vec1 = int(id1)
                        last_vec2 = int(id2)
                else:
                    continue
        print(last_dis)
        mid_post_list = []
        mid_post_list.append(last_vec1)
        mid_post_list.append(last_vec2)
        return mid_post_list

    def top_dis_entity_manhattan(self,mid_post_index, embedding):
        top_dis = 0
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    vec_sub = vec1-vec2
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    if manhattan_sim.item() > top_dis:
                        top_dis = manhattan_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def top_dis_5entity_cos(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    top_dis.append(cos_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[cos_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list

    def top_dis_5entity_manhattan(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    vec_sub = vec1 - vec2
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    top_dis.append(manhattan_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[manhattan_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        sim_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])
                sim_post_list.append(sim)

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list, sim_post_list

    def pad_eneity(self,mid,mid2index):
        if len(mid) < self.entity_len:
            pad_len = self.entity_len-len(mid)
            for j in range(pad_len):
                mid.append(mid2index["<pad{}>".format(j+1)])
        # assert len(mid) == self.entity_len
        # print(mid)
        return mid

    def img_trans(self):
        #定义一个变量transform，是对输入图像进行的所有图像转换的组合
        transform = transforms.Compose([
            transforms.Resize(256),
            #图像调整为256x256像素
            transforms.CenterCrop(224),
            #将图像中心裁剪出来，大小为224x224像素
            transforms.ToTensor(),
            #图像转换为tensor数据类型
            #将图像的平均值和标准差设置为指定的值来正则化图像
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )])
        return transform

    def load_data(self):
        #对bert处理
        # segments = []
        attention_masks_train = []
        input_ids_train = []
        tokens_train = self.sentences_train

        attention_masks_test = []
        input_ids_test = []
        tokens_test = self.sentences_test

        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in tokens_train:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids_train.append(tkn['input_ids'].squeeze(0))
            attention_masks_train.append(tkn['attention_mask'].squeeze(0))

        for tkn in tokens_test:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids_test.append(tkn['input_ids'].squeeze(0))
            attention_masks_test.append(tkn['attention_mask'].squeeze(0))
        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        #对label处理
        y_train = [int(label) for label in self.y_train]
        y_train = torch.LongTensor(y_train)

        y_test = [int(label) for label in self.y_test]
        y_test = torch.LongTensor(y_test)

        #对img_label处理
        y_img_train = [int(label) for label in self.y_img_train]
        y_img_train = torch.LongTensor(y_img_train)

        y_img_test = [int(label) for label in self.y_img_test]
        y_img_test = torch.LongTensor(y_img_test)

        #对实体mids进行处理
        mid2index = self.make_dic()
        embedding = self.get_transe()  # torch.Size([86054151, 50])
        # print('transe size:',embedding.size())
        # count = 0
        vec_all_1_train, vec_all_2_train, sim_all_train = [],[],[]

        for i, mid_post in enumerate(self.mids_train):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1_train.append(vec_post_1)
            vec_all_2_train.append(vec_post_2)
            sim_all_train.append(sim_post)
            # print(i)
        # return torch.LongTensor(vec_all)
        # print("vec_all_1",vec_all_1)
        # print("vec_all_2",vec_all_2)
        vec_all_1_test, vec_all_2_test, sim_all_test = [], [], []

        for i, mid_post in enumerate(self.mids_test):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1_test.append(vec_post_1)
            vec_all_2_test.append(vec_post_2)
            sim_all_test.append(sim_post)
        #train\val\test

        fold_val_0, fold_test_0 = split_data(len(y_test), y_test,self.config.val_en,\
                                                              self.config.test_en, shuffle=True)
        fold_val_1, fold_test_1 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)
        fold_val_2, fold_test_2 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)
        fold_val_3, fold_test_3 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)
        fold_val_4, fold_test_4 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)


        train_dict_0, val_dict_0, test_dict_0 = {}, {}, {}
        train_dict_1, val_dict_1, test_dict_1 = {}, {}, {}
        train_dict_2, val_dict_2, test_dict_2 = {}, {}, {}
        train_dict_3, val_dict_3, test_dict_3 = {}, {}, {}
        train_dict_4, val_dict_4, test_dict_4 = {}, {}, {}

        names_dict_test = {'input_ids':input_ids_test,'attention_masks':attention_masks_test,'image':self.images_test,\
                      'vec_1':vec_all_1_test, 'vec_2':vec_all_2_test, 'sim_list':sim_all_test,'y':y_test,'y_img':y_img_test}
        for name in names_dict_test:
            # train_dict_0[name] = [names_dict[name][i] for i in fold0_train]
            val_dict_0[name] = [names_dict_test[name][i] for i in fold_val_0]
            val_dict_1[name] = [names_dict_test[name][i] for i in fold_val_1]
            val_dict_2[name] = [names_dict_test[name][i] for i in fold_val_2]
            val_dict_3[name] = [names_dict_test[name][i] for i in fold_val_3]
            val_dict_4[name] = [names_dict_test[name][i] for i in fold_val_4]

            test_dict_0[name] = [names_dict_test[name][i] for i in fold_test_0]
            test_dict_1[name] = [names_dict_test[name][i] for i in fold_test_1]
            test_dict_2[name] = [names_dict_test[name][i] for i in fold_test_2]
            test_dict_3[name] = [names_dict_test[name][i] for i in fold_test_3]
            test_dict_4[name] = [names_dict_test[name][i] for i in fold_test_4]

        names_dict_train = {'input_ids': input_ids_train, 'attention_masks': attention_masks_train,
                            'image': self.images_train, \
                            'vec_1': vec_all_1_train, 'vec_2': vec_all_2_train, 'sim_list': sim_all_train, 'y': y_train,
                            'y_img': y_img_train}
        for name in names_dict_train:
            train_dict_0[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_1[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_2[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_3[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_4[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            # val_dict[name] = [names_dict_train[name][i] for i in fold_val]
            # test_dict[name] = [names_dict_train[name][i] for i in fold_test]



        return train_dict_0, val_dict_0, test_dict_0, \
               train_dict_1, val_dict_1, test_dict_1, \
               train_dict_2, val_dict_2, test_dict_2, \
               train_dict_3, val_dict_3, test_dict_3, \
               train_dict_4, val_dict_4, test_dict_4

#weibo\pheme 伪实体补充random 变为seed
class data_preprocess_ATT_bert_nfold_random():
    def __init__(self,sentences,images,mids,y,y_img,config,pathset,args): #sentence==token_ids
        #glove_path = '/home/sunmengzhu2019/kg_rumor/w2v/merge_sgns_bigram_char300.txt',
        #transE_path = '/home/sunmengzhu2019/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        # self.glove_path = pathset.path_glove
        self.sentences = sentences
        self.images = images
        self.mids = mids
        self.y = y
        self.y_img = y_img
        self.args = args

        self.sen_len = config.sen_len
        self.entity_len = config.entity_len

        self.transE_path = pathset.path_transe
        self.pathset = pathset
        self.config = config

        self.index2mid = []
        self.mid2index = {}
        self.index2word = []
        self.word2index = {}
        # self.embedding_matrix = []
        # self.embedding_glove = []

    def get_transe(self):
        vec = np.memmap(self.transE_path, dtype='float32', mode='r')
        # row = int(vec.shape[0]/50)
        self.embedding = vec.reshape((-1,50))
        self.embedding_entity_dim = 50 #50

        self.add_entity_embedding("<pad1>",2)
        self.add_entity_embedding("<pad2>",3)
        self.add_entity_embedding("<pad3>",4)
        self.add_entity_embedding("<pad4>",5)
        # #归一化
        # min_max_scaler = MinMaxScaler()
        # self.embedding_minmax = min_max_scaler.fit_transform(self.embedding)
        #标准化
        mu = np.mean(self.embedding, axis=0)
        std = np.std(self.embedding, axis=0)
        self.embedding_mnstd = (self.embedding - mu) / std
        self.embedding_fin = torch.from_numpy(self.embedding_mnstd)
        return torch.tensor(self.embedding_fin)
        # return torch.tensor(self.embedding)

    def make_dic(self):
        dic_path = self.pathset.path_dic
        with open(dic_path,'r',encoding='utf-8')as t:
            text_all = t.readlines()
        for line in text_all[1:]:
            mid = line.split('\t')[0]
            idx = line.split('\t')[1].strip('\n')
            self.mid2index[mid] = int(idx)
        return self.mid2index

#TODO:random的问题
    def add_entity_embedding(self,entity,num):
        # vec_entity = torch.empty(1, 50)
        # nn.init.uniform_(vec_entity)

        #seed
        torch.manual_seed(num)
        vec_entity = torch.randn(1,50)
        self.mid2index[entity] = len(self.mid2index)
        # self.index2mid.append(entity)
        vec_entity = vec_entity.numpy()
        self.embedding = np.concatenate([self.embedding, vec_entity], 0)

    def top_dis_entity_cos(self,mid_post_index,embedding):
        # top_dis = 0
        top_dis = -1
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() > top_dis:
                        top_dis = cos_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def last_dis_entity_cos(self,mid_post_index,embedding):
        last_dis = 1
        last_vec1 = 0
        last_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() < last_dis:
                        last_dis = cos_sim.item()
                        last_vec1 = int(id1)
                        last_vec2 = int(id2)
                else:
                    continue
        print(last_dis)
        mid_post_list = []
        mid_post_list.append(last_vec1)
        mid_post_list.append(last_vec2)
        return mid_post_list

    def top_dis_entity_manhattan(self,mid_post_index, embedding):
        top_dis = 0
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    vec_sub = vec1-vec2
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    if manhattan_sim.item() > top_dis:
                        top_dis = manhattan_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def top_dis_5entity_cos(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    top_dis.append(cos_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[cos_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list

    def top_dis_5entity_manhattan(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    vec_sub = vec1 - vec2
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    top_dis.append(manhattan_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[manhattan_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        sim_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])
                sim_post_list.append(sim)

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list, sim_post_list

    def pad_eneity(self,mid,mid2index):
        if len(mid) < self.entity_len:
            pad_len = self.entity_len-len(mid)
            for j in range(pad_len):
                mid.append(mid2index["<pad{}>".format(j+1)])
        # assert len(mid) == self.entity_len
        # print(mid)
        return mid

    def img_trans(self):
        #定义一个变量transform，是对输入图像进行的所有图像转换的组合
        transform = transforms.Compose([
            transforms.Resize(256),
            #图像调整为256x256像素
            transforms.CenterCrop(224),
            #将图像中心裁剪出来，大小为224x224像素
            transforms.ToTensor(),
            #图像转换为tensor数据类型
            #将图像的平均值和标准差设置为指定的值来正则化图像
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )])
        return transform

    def load_data(self):
        #对bert处理
        # segments = []
        attention_masks = []
        input_ids = []
        tokens = self.sentences
        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in tokens:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids.append(tkn['input_ids'].squeeze(0))
            attention_masks.append(tkn['attention_mask'].squeeze(0))

        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        #对label处理
        y = [int(label) for label in self.y]
        y = torch.LongTensor(y)

        #对img_label处理
        y_img = [int(label) for label in self.y_img]
        y_img = torch.LongTensor(y_img)

        #对实体mids进行处理
        mid2index = self.make_dic()
        embedding = self.get_transe()  # torch.Size([86054151, 50])
        # print('transe size:',embedding.size())
        # count = 0
        vec_all_1, vec_all_2, sim_all = [],[],[]

        for i, mid_post in enumerate(self.mids):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1.append(vec_post_1)
            vec_all_2.append(vec_post_2)
            sim_all.append(sim_post)
            # print(i)
        # return torch.LongTensor(vec_all)
        # print("vec_all_1",vec_all_1)
        # print("vec_all_2",vec_all_2)

        #train\val\test
        fold0_test, fold0_val, fold0_train, \
        fold1_test, fold1_val, fold1_train, \
        fold2_test, fold2_val, fold2_train, \
        fold3_test, fold3_val, fold3_train, \
        fold4_test, fold4_val, fold4_train = split_data_5fold(len(y), y,self.config.train,self.config.val,\
                                                              self.config.test, shuffle=True)

        train_dict_0, val_dict_0, test_dict_0 = {}, {}, {}
        train_dict_1, val_dict_1, test_dict_1 = {}, {}, {}
        train_dict_2, val_dict_2, test_dict_2 = {}, {}, {}
        train_dict_3, val_dict_3, test_dict_3 = {}, {}, {}
        train_dict_4, val_dict_4, test_dict_4 = {}, {}, {}

        names_dict = {'input_ids':input_ids,'attention_masks':attention_masks,'image':self.images,\
                      'vec_1':vec_all_1, 'vec_2':vec_all_2, 'sim_list':sim_all,'y':y,'y_img':y_img}
        for name in names_dict:
            train_dict_0[name] = [names_dict[name][i] for i in fold0_train]
            val_dict_0[name] = [names_dict[name][i] for i in fold0_val]
            test_dict_0[name] = [names_dict[name][i] for i in fold0_test]
        for name in names_dict:
            train_dict_1[name] = [names_dict[name][i] for i in fold1_train]
            val_dict_1[name] = [names_dict[name][i] for i in fold1_val]
            test_dict_1[name] = [names_dict[name][i] for i in fold1_test]
        for name in names_dict:
            train_dict_2[name] = [names_dict[name][i] for i in fold2_train]
            val_dict_2[name] = [names_dict[name][i] for i in fold2_val]
            test_dict_2[name] = [names_dict[name][i] for i in fold2_test]
        for name in names_dict:
            train_dict_3[name] = [names_dict[name][i] for i in fold3_train]
            val_dict_3[name] = [names_dict[name][i] for i in fold3_val]
            test_dict_3[name] = [names_dict[name][i] for i in fold3_test]
        for name in names_dict:
            train_dict_4[name] = [names_dict[name][i] for i in fold4_train]
            val_dict_4[name] = [names_dict[name][i] for i in fold4_val]
            test_dict_4[name] = [names_dict[name][i] for i in fold4_test]



        return train_dict_0, val_dict_0, test_dict_0, \
               train_dict_1, val_dict_1, test_dict_1, \
               train_dict_2, val_dict_2, test_dict_2, \
               train_dict_3, val_dict_3, test_dict_3, \
               train_dict_4, val_dict_4, test_dict_4

#twitter 伪实体补充random 变为seed
class data_preprocess_ATT_bert_random():
    def __init__(self,sentences_train,images_train,mids_train,y_train,y_img_train,\
                 sentences_test,images_test,mids_test,y_test,y_img_test,config,pathset,args): #sentence==token_ids
        #glove_path = '/home/sunmengzhu2019/kg_rumor/w2v/merge_sgns_bigram_char300.txt',
        #transE_path = '/home/sunmengzhu2019/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        # self.glove_path = pathset.path_glove
        self.sentences_train = sentences_train
        self.images_train = images_train
        self.mids_train = mids_train
        self.y_train = y_train
        self.y_img_train = y_img_train

        self.sentences_test = sentences_test
        self.images_test = images_test
        self.mids_test = mids_test
        self.y_test = y_test
        self.y_img_test = y_img_test

        self.sen_len = config.sen_len
        self.entity_len = config.entity_len

        self.transE_path = pathset.path_transe
        self.pathset = pathset
        self.config = config

        self.index2mid = []
        self.mid2index = {}
        self.index2word = []
        self.word2index = {}
        # self.embedding_matrix = []
        # self.embedding_glove = []

    def get_transe(self):
        vec = np.memmap(self.transE_path, dtype='float32', mode='r')
        # row = int(vec.shape[0]/50)
        self.embedding = vec.reshape((-1,50))
        self.embedding_entity_dim = 50 #50

        self.add_entity_embedding("<pad1>",2)
        self.add_entity_embedding("<pad2>",3)
        self.add_entity_embedding("<pad3>",4)
        self.add_entity_embedding("<pad4>",5)
        # #归一化
        # min_max_scaler = MinMaxScaler()
        # self.embedding_minmax = min_max_scaler.fit_transform(self.embedding)
        #标准化
        mu = np.mean(self.embedding, axis=0)
        std = np.std(self.embedding, axis=0)
        self.embedding_mnstd = (self.embedding - mu) / std
        self.embedding_fin = torch.from_numpy(self.embedding_mnstd)
        return torch.tensor(self.embedding_fin)
        # return torch.tensor(self.embedding)

    def make_dic(self):
        dic_path = self.pathset.path_dic
        with open(dic_path,'r',encoding='utf-8')as t:
            text_all = t.readlines()
        for line in text_all[1:]:
            mid = line.split('\t')[0]
            idx = line.split('\t')[1].strip('\n')
            self.mid2index[mid] = int(idx)
        return self.mid2index

#TODO:RANDOM的问题
    def add_entity_embedding(self,entity,num):
        # vec_entity = torch.empty(1, 50)
        # nn.init.uniform_(vec_entity)
        #seed
        torch.manual_seed(num)
        vec_entity = torch.randn(1,50)
        self.mid2index[entity] = len(self.mid2index)
        # self.index2mid.append(entity)
        vec_entity = vec_entity.numpy()
        self.embedding = np.concatenate([self.embedding, vec_entity], 0)

    def top_dis_entity_cos(self,mid_post_index,embedding):
        # top_dis = 0
        top_dis = -1
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() > top_dis:
                        top_dis = cos_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def last_dis_entity_cos(self,mid_post_index,embedding):
        last_dis = 1
        last_vec1 = 0
        last_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() < last_dis:
                        last_dis = cos_sim.item()
                        last_vec1 = int(id1)
                        last_vec2 = int(id2)
                else:
                    continue
        print(last_dis)
        mid_post_list = []
        mid_post_list.append(last_vec1)
        mid_post_list.append(last_vec2)
        return mid_post_list

    def top_dis_entity_manhattan(self,mid_post_index, embedding):
        top_dis = 0
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    vec_sub = vec1-vec2
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    if manhattan_sim.item() > top_dis:
                        top_dis = manhattan_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def top_dis_5entity_cos(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    top_dis.append(cos_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[cos_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list

    def top_dis_5entity_manhattan(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    vec_sub = vec1 - vec2
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    top_dis.append(manhattan_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[manhattan_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        sim_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])
                sim_post_list.append(sim)

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list, sim_post_list

    def pad_eneity(self,mid,mid2index):
        if len(mid) < self.entity_len:
            pad_len = self.entity_len-len(mid)
            for j in range(pad_len):
                mid.append(mid2index["<pad{}>".format(j+1)])
        # assert len(mid) == self.entity_len
        # print(mid)
        return mid

    def img_trans(self):
        #定义一个变量transform，是对输入图像进行的所有图像转换的组合
        transform = transforms.Compose([
            transforms.Resize(256),
            #图像调整为256x256像素
            transforms.CenterCrop(224),
            #将图像中心裁剪出来，大小为224x224像素
            transforms.ToTensor(),
            #图像转换为tensor数据类型
            #将图像的平均值和标准差设置为指定的值来正则化图像
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )])
        return transform

    def load_data(self):
        #对bert处理
        # segments = []
        attention_masks_train = []
        input_ids_train = []
        tokens_train = self.sentences_train

        attention_masks_test = []
        input_ids_test = []
        tokens_test = self.sentences_test

        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in tokens_train:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids_train.append(tkn['input_ids'].squeeze(0))
            attention_masks_train.append(tkn['attention_mask'].squeeze(0))

        for tkn in tokens_test:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids_test.append(tkn['input_ids'].squeeze(0))
            attention_masks_test.append(tkn['attention_mask'].squeeze(0))
        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        #对label处理
        y_train = [int(label) for label in self.y_train]
        y_train = torch.LongTensor(y_train)

        y_test = [int(label) for label in self.y_test]
        y_test = torch.LongTensor(y_test)

        #对img_label处理
        y_img_train = [int(label) for label in self.y_img_train]
        y_img_train = torch.LongTensor(y_img_train)

        y_img_test = [int(label) for label in self.y_img_test]
        y_img_test = torch.LongTensor(y_img_test)

        #对实体mids进行处理
        mid2index = self.make_dic()
        embedding = self.get_transe()  # torch.Size([86054151, 50])
        # print('transe size:',embedding.size())
        # count = 0
        vec_all_1_train, vec_all_2_train, sim_all_train = [],[],[]

        for i, mid_post in enumerate(self.mids_train):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1_train.append(vec_post_1)
            vec_all_2_train.append(vec_post_2)
            sim_all_train.append(sim_post)
            # print(i)
        # return torch.LongTensor(vec_all)
        # print("vec_all_1",vec_all_1)
        # print("vec_all_2",vec_all_2)
        vec_all_1_test, vec_all_2_test, sim_all_test = [], [], []

        for i, mid_post in enumerate(self.mids_test):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1_test.append(vec_post_1)
            vec_all_2_test.append(vec_post_2)
            sim_all_test.append(sim_post)
        #train\val\test

        fold_val_0, fold_test_0 = split_data(len(y_test), y_test,self.config.val_en,\
                                                              self.config.test_en, shuffle=True)
        fold_val_1, fold_test_1 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)
        fold_val_2, fold_test_2 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)
        fold_val_3, fold_test_3 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)
        fold_val_4, fold_test_4 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)


        train_dict_0, val_dict_0, test_dict_0 = {}, {}, {}
        train_dict_1, val_dict_1, test_dict_1 = {}, {}, {}
        train_dict_2, val_dict_2, test_dict_2 = {}, {}, {}
        train_dict_3, val_dict_3, test_dict_3 = {}, {}, {}
        train_dict_4, val_dict_4, test_dict_4 = {}, {}, {}

        names_dict_test = {'input_ids':input_ids_test,'attention_masks':attention_masks_test,'image':self.images_test,\
                      'vec_1':vec_all_1_test, 'vec_2':vec_all_2_test, 'sim_list':sim_all_test,'y':y_test,'y_img':y_img_test}
        for name in names_dict_test:
            # train_dict_0[name] = [names_dict[name][i] for i in fold0_train]
            val_dict_0[name] = [names_dict_test[name][i] for i in fold_val_0]
            val_dict_1[name] = [names_dict_test[name][i] for i in fold_val_1]
            val_dict_2[name] = [names_dict_test[name][i] for i in fold_val_2]
            val_dict_3[name] = [names_dict_test[name][i] for i in fold_val_3]
            val_dict_4[name] = [names_dict_test[name][i] for i in fold_val_4]

            test_dict_0[name] = [names_dict_test[name][i] for i in fold_test_0]
            test_dict_1[name] = [names_dict_test[name][i] for i in fold_test_1]
            test_dict_2[name] = [names_dict_test[name][i] for i in fold_test_2]
            test_dict_3[name] = [names_dict_test[name][i] for i in fold_test_3]
            test_dict_4[name] = [names_dict_test[name][i] for i in fold_test_4]

        names_dict_train = {'input_ids': input_ids_train, 'attention_masks': attention_masks_train,
                            'image': self.images_train, \
                            'vec_1': vec_all_1_train, 'vec_2': vec_all_2_train, 'sim_list': sim_all_train, 'y': y_train,
                            'y_img': y_img_train}
        for name in names_dict_train:
            train_dict_0[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_1[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_2[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_3[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_4[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            # val_dict[name] = [names_dict_train[name][i] for i in fold_val]
            # test_dict[name] = [names_dict_train[name][i] for i in fold_test]



        return train_dict_0, val_dict_0, test_dict_0, \
               train_dict_1, val_dict_1, test_dict_1, \
               train_dict_2, val_dict_2, test_dict_2, \
               train_dict_3, val_dict_3, test_dict_3, \
               train_dict_4, val_dict_4, test_dict_4


# weibo\pheme 伪实体补充random 变为seed 适用于全图数据集
class data_preprocess_ATT_bert_nfold_random_full_image():
    def __init__(self, sentences, images, mids, y, y_img, config, pathset, args):  # sentence==token_ids
        # glove_path = '/home/sunmengzhu2019/kg_rumor/w2v/merge_sgns_bigram_char300.txt',
        # transE_path = '/home/sunmengzhu2019/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        # self.glove_path = pathset.path_glove
        self.sentences = sentences
        self.images = images
        self.mids = mids
        self.y = y
        self.y_img = y_img
        self.args = args

        self.sen_len = config.sen_len
        self.entity_len = config.entity_len

        self.transE_path = pathset.path_transe
        self.pathset = pathset
        self.config = config

        self.index2mid = []
        self.mid2index = {}
        self.index2word = []
        self.word2index = {}
        # self.embedding_matrix = []
        # self.embedding_glove = []

    def get_transe(self):
        vec = np.memmap(self.transE_path, dtype='float32', mode='r')
        # row = int(vec.shape[0]/50)
        self.embedding = vec.reshape((-1, 50))
        self.embedding_entity_dim = 50  # 50

        self.add_entity_embedding("<pad1>", 2)
        self.add_entity_embedding("<pad2>", 3)
        self.add_entity_embedding("<pad3>", 4)
        self.add_entity_embedding("<pad4>", 5)
        # #归一化
        # min_max_scaler = MinMaxScaler()
        # self.embedding_minmax = min_max_scaler.fit_transform(self.embedding)
        # 标准化
        mu = np.mean(self.embedding, axis=0)
        std = np.std(self.embedding, axis=0)
        self.embedding_mnstd = (self.embedding - mu) / std
        self.embedding_fin = torch.from_numpy(self.embedding_mnstd)
        return torch.tensor(self.embedding_fin)
        # return torch.tensor(self.embedding)

    def make_dic(self):
        dic_path = self.pathset.path_dic
        with open(dic_path, 'r', encoding='utf-8') as t:
            text_all = t.readlines()
        for line in text_all[1:]:
            mid = line.split('\t')[0]
            idx = line.split('\t')[1].strip('\n')
            self.mid2index[mid] = int(idx)
        return self.mid2index

    # TODO:random的问题
    def add_entity_embedding(self, entity, num):
        # vec_entity = torch.empty(1, 50)
        # nn.init.uniform_(vec_entity)

        # seed
        torch.manual_seed(num)
        vec_entity = torch.randn(1, 50)
        self.mid2index[entity] = len(self.mid2index)
        # self.index2mid.append(entity)
        vec_entity = vec_entity.numpy()
        self.embedding = np.concatenate([self.embedding, vec_entity], 0)

    def top_dis_entity_cos(self, mid_post_index, embedding):
        # top_dis = 0
        top_dis = -1
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    if cos_sim.item() > top_dis:
                        top_dis = cos_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def last_dis_entity_cos(self, mid_post_index, embedding):
        last_dis = 1
        last_vec1 = 0
        last_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    if cos_sim.item() < last_dis:
                        last_dis = cos_sim.item()
                        last_vec1 = int(id1)
                        last_vec2 = int(id2)
                else:
                    continue
        print(last_dis)
        mid_post_list = []
        mid_post_list.append(last_vec1)
        mid_post_list.append(last_vec2)
        return mid_post_list

    def top_dis_entity_manhattan(self, mid_post_index, embedding):
        top_dis = 0
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    vec_sub = vec1 - vec2
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    if manhattan_sim.item() > top_dis:
                        top_dis = manhattan_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def top_dis_5entity_cos(self, mid_post_index, embedding, mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index, mid2index)  # len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                            torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    top_dis.append(cos_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[cos_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        for sim in sorted(dis_vec, reverse=True):  # sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list

    def top_dis_5entity_manhattan(self, mid_post_index, embedding, mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index, mid2index)  # len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    vec_sub = vec1 - vec2
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    top_dis.append(manhattan_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[manhattan_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        sim_post_list = []
        for sim in sorted(dis_vec, reverse=True):  # sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])
                sim_post_list.append(sim)

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list, sim_post_list

    def pad_eneity(self, mid, mid2index):
        if len(mid) < self.entity_len:
            pad_len = self.entity_len - len(mid)
            for j in range(pad_len):
                mid.append(mid2index["<pad{}>".format(j + 1)])
        # assert len(mid) == self.entity_len
        # print(mid)
        return mid

    def img_trans(self):
        # 定义一个变量transform，是对输入图像进行的所有图像转换的组合
        transform = transforms.Compose([
            transforms.Resize(256),
            # 图像调整为256x256像素
            transforms.CenterCrop(224),
            # 将图像中心裁剪出来，大小为224x224像素
            transforms.ToTensor(),
            # 图像转换为tensor数据类型
            # 将图像的平均值和标准差设置为指定的值来正则化图像
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        return transform

    def load_data(self):
        # 对bert处理
        # segments = []
        attention_masks = []
        input_ids = []
        tokens = self.sentences
        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in tokens:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids.append(tkn['input_ids'].squeeze(0))
            attention_masks.append(tkn['attention_mask'].squeeze(0))

        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        # 对label处理
        y = [int(label) for label in self.y]
        y = torch.LongTensor(y)

        # 对img_label处理
        y_img = [int(label) for label in self.y_img]
        y_img = torch.LongTensor(y_img)

        # 对实体mids进行处理
        mid2index = self.make_dic()
        embedding = self.get_transe()  # torch.Size([86054151, 50])
        # print('transe size:',embedding.size())
        # count = 0
        vec_all_1, vec_all_2, sim_all = [], [], []

        for i, mid_post in enumerate(self.mids):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1.append(vec_post_1)
            vec_all_2.append(vec_post_2)
            sim_all.append(sim_post)
            # print(i)
        # return torch.LongTensor(vec_all)
        # print("vec_all_1",vec_all_1)
        # print("vec_all_2",vec_all_2)

        # train\val\test
        fold0_test, fold0_val, fold0_train, \
        fold1_test, fold1_val, fold1_train, \
        fold2_test, fold2_val, fold2_train, \
        fold3_test, fold3_val, fold3_train, \
        fold4_test, fold4_val, fold4_train = split_data_5fold(len(y), y, self.config.train, self.config.val, \
                                                              self.config.test, shuffle=True)

        train_dict_0, val_dict_0, test_dict_0 = {}, {}, {}
        train_dict_1, val_dict_1, test_dict_1 = {}, {}, {}
        train_dict_2, val_dict_2, test_dict_2 = {}, {}, {}
        train_dict_3, val_dict_3, test_dict_3 = {}, {}, {}
        train_dict_4, val_dict_4, test_dict_4 = {}, {}, {}

        names_dict = {'input_ids': input_ids, 'attention_masks': attention_masks, 'image': self.images, \
                      'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img}
        for name in names_dict:
            train_dict_0[name] = [names_dict[name][i] for i in fold0_train]
            val_dict_0[name] = [names_dict[name][i] for i in fold0_val]
            test_dict_0[name] = [names_dict[name][i] for i in fold0_test]
        for name in names_dict:
            train_dict_1[name] = [names_dict[name][i] for i in fold1_train]
            val_dict_1[name] = [names_dict[name][i] for i in fold1_val]
            test_dict_1[name] = [names_dict[name][i] for i in fold1_test]
        for name in names_dict:
            train_dict_2[name] = [names_dict[name][i] for i in fold2_train]
            val_dict_2[name] = [names_dict[name][i] for i in fold2_val]
            test_dict_2[name] = [names_dict[name][i] for i in fold2_test]
        for name in names_dict:
            train_dict_3[name] = [names_dict[name][i] for i in fold3_train]
            val_dict_3[name] = [names_dict[name][i] for i in fold3_val]
            test_dict_3[name] = [names_dict[name][i] for i in fold3_test]
        for name in names_dict:
            train_dict_4[name] = [names_dict[name][i] for i in fold4_train]
            val_dict_4[name] = [names_dict[name][i] for i in fold4_val]
            test_dict_4[name] = [names_dict[name][i] for i in fold4_test]

        #保存npy格式数据，之后的实验直接加载.npy文件 直接存取字典
        # embedding = np.array(embedding)
        # # 保存npy文件
        # with open(clean_image_dir + '/image.npy', 'wb') as f:
        #     # root_index_final = np.array(root_index_final)
        #     np.save(f, embedding)
        # # print('finish')
        train_dict_list = [train_dict_0,train_dict_1,train_dict_2,train_dict_3,train_dict_4]
        val_dict_list = [val_dict_0, val_dict_1, val_dict_2, val_dict_3, val_dict_4]
        test_dict_list = [test_dict_0, test_dict_1, test_dict_2, test_dict_3, test_dict_4]

        save_split_path = '../data/{}/full_image/'.format(self.args.dataset)
        if not os.path.exists(save_split_path):
            os.mkdir(save_split_path)
        for i in range(5):
            np.save(save_split_path+'{}_split_train_{}'.format(self.args.dataset,i),train_dict_list[i])
            np.save(save_split_path + '{}_split_val_{}'.format(self.args.dataset, i), val_dict_list[i])
            np.save(save_split_path + '{}_split_test_{}'.format(self.args.dataset, i), test_dict_list[i])
        #npy文件读取
        # dict = {'a':{1,2,3},'b':{4,5,6}}
        # dict_load = np.load('xxx.npy',allow_pickle=True)
        # dict_load.item()
        # dict_load.item()['a']


        return train_dict_0, val_dict_0, test_dict_0, \
               train_dict_1, val_dict_1, test_dict_1, \
               train_dict_2, val_dict_2, test_dict_2, \
               train_dict_3, val_dict_3, test_dict_3, \
               train_dict_4, val_dict_4, test_dict_4


# twitter 伪实体补充random 变为seed 适用于全图数据集
class data_preprocess_ATT_bert_random_full_image():
    def __init__(self, sentences_train, images_train, mids_train, y_train, y_img_train, \
                 sentences_test, images_test, mids_test, y_test, y_img_test, config, pathset,
                 args):  # sentence==token_ids
        # glove_path = '/home/sunmengzhu2019/kg_rumor/w2v/merge_sgns_bigram_char300.txt',
        # transE_path = '/home/sunmengzhu2019/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        # self.glove_path = pathset.path_glove
        self.sentences_train = sentences_train
        self.images_train = images_train
        self.mids_train = mids_train
        self.y_train = y_train
        self.y_img_train = y_img_train

        self.sentences_test = sentences_test
        self.images_test = images_test
        self.mids_test = mids_test
        self.y_test = y_test
        self.y_img_test = y_img_test

        self.sen_len = config.sen_len
        self.entity_len = config.entity_len

        self.transE_path = pathset.path_transe
        self.pathset = pathset
        self.config = config

        self.index2mid = []
        self.mid2index = {}
        self.index2word = []
        self.word2index = {}
        # self.embedding_matrix = []
        # self.embedding_glove = []
        self.args = args

    def get_transe(self):
        vec = np.memmap(self.transE_path, dtype='float32', mode='r')
        # row = int(vec.shape[0]/50)
        self.embedding = vec.reshape((-1, 50))
        self.embedding_entity_dim = 50  # 50

        self.add_entity_embedding("<pad1>", 2)
        self.add_entity_embedding("<pad2>", 3)
        self.add_entity_embedding("<pad3>", 4)
        self.add_entity_embedding("<pad4>", 5)
        # #归一化
        # min_max_scaler = MinMaxScaler()
        # self.embedding_minmax = min_max_scaler.fit_transform(self.embedding)
        # 标准化
        mu = np.mean(self.embedding, axis=0)
        std = np.std(self.embedding, axis=0)
        self.embedding_mnstd = (self.embedding - mu) / std
        self.embedding_fin = torch.from_numpy(self.embedding_mnstd)
        return torch.tensor(self.embedding_fin)
        # return torch.tensor(self.embedding)

    def make_dic(self):
        dic_path = self.pathset.path_dic
        with open(dic_path, 'r', encoding='utf-8') as t:
            text_all = t.readlines()
        for line in text_all[1:]:
            mid = line.split('\t')[0]
            idx = line.split('\t')[1].strip('\n')
            self.mid2index[mid] = int(idx)
        return self.mid2index

    # TODO:RANDOM的问题
    def add_entity_embedding(self, entity, num):
        # vec_entity = torch.empty(1, 50)
        # nn.init.uniform_(vec_entity)
        # seed
        torch.manual_seed(num)
        vec_entity = torch.randn(1, 50)
        self.mid2index[entity] = len(self.mid2index)
        # self.index2mid.append(entity)
        vec_entity = vec_entity.numpy()
        self.embedding = np.concatenate([self.embedding, vec_entity], 0)

    def top_dis_entity_cos(self, mid_post_index, embedding):
        # top_dis = 0
        top_dis = -1
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    if cos_sim.item() > top_dis:
                        top_dis = cos_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def last_dis_entity_cos(self, mid_post_index, embedding):
        last_dis = 1
        last_vec1 = 0
        last_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    if cos_sim.item() < last_dis:
                        last_dis = cos_sim.item()
                        last_vec1 = int(id1)
                        last_vec2 = int(id2)
                else:
                    continue
        print(last_dis)
        mid_post_list = []
        mid_post_list.append(last_vec1)
        mid_post_list.append(last_vec2)
        return mid_post_list

    def top_dis_entity_manhattan(self, mid_post_index, embedding):
        top_dis = 0
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    vec_sub = vec1 - vec2
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    if manhattan_sim.item() > top_dis:
                        top_dis = manhattan_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def top_dis_5entity_cos(self, mid_post_index, embedding, mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index, mid2index)  # len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                            torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    top_dis.append(cos_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[cos_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        for sim in sorted(dis_vec, reverse=True):  # sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list

    def top_dis_5entity_manhattan(self, mid_post_index, embedding, mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index, mid2index)  # len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    vec_sub = vec1 - vec2
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    top_dis.append(manhattan_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[manhattan_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        sim_post_list = []
        for sim in sorted(dis_vec, reverse=True):  # sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])
                sim_post_list.append(sim)

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list, sim_post_list

    def pad_eneity(self, mid, mid2index):
        if len(mid) < self.entity_len:
            pad_len = self.entity_len - len(mid)
            for j in range(pad_len):
                mid.append(mid2index["<pad{}>".format(j + 1)])
        # assert len(mid) == self.entity_len
        # print(mid)
        return mid

    def img_trans(self):
        # 定义一个变量transform，是对输入图像进行的所有图像转换的组合
        transform = transforms.Compose([
            transforms.Resize(256),
            # 图像调整为256x256像素
            transforms.CenterCrop(224),
            # 将图像中心裁剪出来，大小为224x224像素
            transforms.ToTensor(),
            # 图像转换为tensor数据类型
            # 将图像的平均值和标准差设置为指定的值来正则化图像
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        return transform

    def load_data(self):
        # 对bert处理
        # segments = []
        attention_masks_train = []
        input_ids_train = []
        tokens_train = self.sentences_train

        attention_masks_test = []
        input_ids_test = []
        tokens_test = self.sentences_test

        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in tokens_train:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids_train.append(tkn['input_ids'].squeeze(0))
            attention_masks_train.append(tkn['attention_mask'].squeeze(0))

        for tkn in tokens_test:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids_test.append(tkn['input_ids'].squeeze(0))
            attention_masks_test.append(tkn['attention_mask'].squeeze(0))
        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        # 对label处理
        y_train = [int(label) for label in self.y_train]
        y_train = torch.LongTensor(y_train)

        y_test = [int(label) for label in self.y_test]
        y_test = torch.LongTensor(y_test)

        # 对img_label处理
        y_img_train = [int(label) for label in self.y_img_train]
        y_img_train = torch.LongTensor(y_img_train)

        y_img_test = [int(label) for label in self.y_img_test]
        y_img_test = torch.LongTensor(y_img_test)

        # 对实体mids进行处理
        mid2index = self.make_dic()
        embedding = self.get_transe()  # torch.Size([86054151, 50])
        # print('transe size:',embedding.size())
        # count = 0
        vec_all_1_train, vec_all_2_train, sim_all_train = [], [], []

        for i, mid_post in enumerate(self.mids_train):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1_train.append(vec_post_1)
            vec_all_2_train.append(vec_post_2)
            sim_all_train.append(sim_post)
            # print(i)
        # return torch.LongTensor(vec_all)
        # print("vec_all_1",vec_all_1)
        # print("vec_all_2",vec_all_2)
        vec_all_1_test, vec_all_2_test, sim_all_test = [], [], []

        for i, mid_post in enumerate(self.mids_test):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1_test.append(vec_post_1)
            vec_all_2_test.append(vec_post_2)
            sim_all_test.append(sim_post)
        # train\val\test

        fold_val_0, fold_test_0 = split_data(len(y_test), y_test, self.config.val_en, \
                                             self.config.test_en, shuffle=True)
        fold_val_1, fold_test_1 = split_data(len(y_test), y_test, self.config.val_en, \
                                             self.config.test_en, shuffle=True)
        fold_val_2, fold_test_2 = split_data(len(y_test), y_test, self.config.val_en, \
                                             self.config.test_en, shuffle=True)
        fold_val_3, fold_test_3 = split_data(len(y_test), y_test, self.config.val_en, \
                                             self.config.test_en, shuffle=True)
        fold_val_4, fold_test_4 = split_data(len(y_test), y_test, self.config.val_en, \
                                             self.config.test_en, shuffle=True)

        train_dict_0, val_dict_0, test_dict_0 = {}, {}, {}
        train_dict_1, val_dict_1, test_dict_1 = {}, {}, {}
        train_dict_2, val_dict_2, test_dict_2 = {}, {}, {}
        train_dict_3, val_dict_3, test_dict_3 = {}, {}, {}
        train_dict_4, val_dict_4, test_dict_4 = {}, {}, {}

        names_dict_test = {'input_ids': input_ids_test, 'attention_masks': attention_masks_test,
                           'image': self.images_test, \
                           'vec_1': vec_all_1_test, 'vec_2': vec_all_2_test, 'sim_list': sim_all_test, 'y': y_test,
                           'y_img': y_img_test}
        for name in names_dict_test:
            # train_dict_0[name] = [names_dict[name][i] for i in fold0_train]
            val_dict_0[name] = [names_dict_test[name][i] for i in fold_val_0]
            val_dict_1[name] = [names_dict_test[name][i] for i in fold_val_1]
            val_dict_2[name] = [names_dict_test[name][i] for i in fold_val_2]
            val_dict_3[name] = [names_dict_test[name][i] for i in fold_val_3]
            val_dict_4[name] = [names_dict_test[name][i] for i in fold_val_4]

            test_dict_0[name] = [names_dict_test[name][i] for i in fold_test_0]
            test_dict_1[name] = [names_dict_test[name][i] for i in fold_test_1]
            test_dict_2[name] = [names_dict_test[name][i] for i in fold_test_2]
            test_dict_3[name] = [names_dict_test[name][i] for i in fold_test_3]
            test_dict_4[name] = [names_dict_test[name][i] for i in fold_test_4]

        names_dict_train = {'input_ids': input_ids_train, 'attention_masks': attention_masks_train,
                            'image': self.images_train, \
                            'vec_1': vec_all_1_train, 'vec_2': vec_all_2_train, 'sim_list': sim_all_train, 'y': y_train,
                            'y_img': y_img_train}
        for name in names_dict_train:
            train_dict_0[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_1[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_2[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_3[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_4[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            # val_dict[name] = [names_dict_train[name][i] for i in fold_val]
            # test_dict[name] = [names_dict_train[name][i] for i in fold_test]

        # 保存npy格式数据，之后的实验直接加载.npy文件 直接存取字典
        # embedding = np.array(embedding)
        # # 保存npy文件
        # with open(clean_image_dir + '/image.npy', 'wb') as f:
        #     # root_index_final = np.array(root_index_final)
        #     np.save(f, embedding)
        # # print('finish')
        train_dict_list = [train_dict_0, train_dict_1, train_dict_2, train_dict_3, train_dict_4]
        val_dict_list = [val_dict_0, val_dict_1, val_dict_2, val_dict_3, val_dict_4]
        test_dict_list = [test_dict_0, test_dict_1, test_dict_2, test_dict_3, test_dict_4]

        save_split_path = '../data/{}/full_image'.format(self.args.dataset)
        if not os.path.exists(save_split_path):
            os.mkdir(save_split_path)
        for i in range(5):
            np.save(save_split_path + '{}_split_train_{}'.format(self.args.dataset, i), train_dict_list[i])
            np.save(save_split_path + '{}_split_val_{}'.format(self.args.dataset, i), val_dict_list[i])
            np.save(save_split_path + '{}_split_test_{}'.format(self.args.dataset, i), test_dict_list[i])
        # npy文件读取
        # dict = {'a':{1,2,3},'b':{4,5,6}}
        # dict_load = np.load('xxx.npy',allow_pickle=True)
        # dict_load.item()
        # dict_load.item()['a']

        return train_dict_0, val_dict_0, test_dict_0, \
               train_dict_1, val_dict_1, test_dict_1, \
               train_dict_2, val_dict_2, test_dict_2, \
               train_dict_3, val_dict_3, test_dict_3, \
               train_dict_4, val_dict_4, test_dict_4

##weibo\pheme 伪实体补充random 变为seed 适用于全图数据集 且对平均每个post所包含实体和平均曼哈顿距离进行分析
class data_preprocess_ATT_bert_nfold_random_full_image_analysis():
    def __init__(self, sentences, images, mids, y, y_img, config, pathset, args):  # sentence==token_ids
        # glove_path = '/home/sunmengzhu2019/kg_rumor/w2v/merge_sgns_bigram_char300.txt',
        # transE_path = '/home/sunmengzhu2019/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        # self.glove_path = pathset.path_glove
        self.sentences = sentences
        self.images = images
        self.mids = mids
        self.y = y
        self.y_img = y_img
        self.args = args

        self.sen_len = config.sen_len
        self.entity_len = config.entity_len

        self.transE_path = pathset.path_transe
        self.pathset = pathset
        self.config = config

        self.index2mid = []
        self.mid2index = {}
        self.index2word = []
        self.word2index = {}
        # self.embedding_matrix = []
        # self.embedding_glove = []

    def get_transe(self):
        vec = np.memmap(self.transE_path, dtype='float32', mode='r')
        # row = int(vec.shape[0]/50)
        self.embedding = vec.reshape((-1, 50))
        self.embedding_entity_dim = 50  # 50

        self.add_entity_embedding("<pad1>", 2)
        self.add_entity_embedding("<pad2>", 3)
        self.add_entity_embedding("<pad3>", 4)
        self.add_entity_embedding("<pad4>", 5)
        # #归一化
        # min_max_scaler = MinMaxScaler()
        # self.embedding_minmax = min_max_scaler.fit_transform(self.embedding)
        # 标准化
        mu = np.mean(self.embedding, axis=0)
        std = np.std(self.embedding, axis=0)
        self.embedding_mnstd = (self.embedding - mu) / std
        self.embedding_fin = torch.from_numpy(self.embedding_mnstd)
        return torch.tensor(self.embedding_fin)
        # return torch.tensor(self.embedding)

    def make_dic(self):
        dic_path = self.pathset.path_dic
        with open(dic_path, 'r', encoding='utf-8') as t:
            text_all = t.readlines()
        for line in text_all[1:]:
            mid = line.split('\t')[0]
            idx = line.split('\t')[1].strip('\n')
            self.mid2index[mid] = int(idx)
        return self.mid2index

    # TODO:random的问题
    def add_entity_embedding(self, entity, num):
        # vec_entity = torch.empty(1, 50)
        # nn.init.uniform_(vec_entity)

        # seed
        torch.manual_seed(num)
        vec_entity = torch.randn(1, 50)
        self.mid2index[entity] = len(self.mid2index)
        # self.index2mid.append(entity)
        vec_entity = vec_entity.numpy()
        self.embedding = np.concatenate([self.embedding, vec_entity], 0)

    def top_dis_entity_cos(self, mid_post_index, embedding):
        # top_dis = 0
        top_dis = -1
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    if cos_sim.item() > top_dis:
                        top_dis = cos_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def last_dis_entity_cos(self, mid_post_index, embedding):
        last_dis = 1
        last_vec1 = 0
        last_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    if cos_sim.item() < last_dis:
                        last_dis = cos_sim.item()
                        last_vec1 = int(id1)
                        last_vec2 = int(id2)
                else:
                    continue
        print(last_dis)
        mid_post_list = []
        mid_post_list.append(last_vec1)
        mid_post_list.append(last_vec2)
        return mid_post_list

    def top_dis_entity_manhattan(self, mid_post_index, embedding):
        top_dis = 0
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    vec_sub = vec1 - vec2
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    if manhattan_sim.item() > top_dis:
                        top_dis = manhattan_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def top_dis_5entity_cos(self, mid_post_index, embedding, mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index, mid2index)  # len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                            torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    top_dis.append(cos_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[cos_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        for sim in sorted(dis_vec, reverse=True):  # sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list

    def top_dis_5entity_manhattan(self, mid_post_index, embedding, mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index, mid2index)  # len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    vec_sub = vec1 - vec2
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    top_dis.append(manhattan_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[manhattan_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        sim_post_list = []
        for sim in sorted(dis_vec, reverse=True):  # sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])
                sim_post_list.append(sim)

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list, sim_post_list

    def pad_eneity(self, mid, mid2index):
        if len(mid) < self.entity_len:
            pad_len = self.entity_len - len(mid)
            for j in range(pad_len):
                mid.append(mid2index["<pad{}>".format(j + 1)])
        # assert len(mid) == self.entity_len
        # print(mid)
        return mid

    def img_trans(self):
        # 定义一个变量transform，是对输入图像进行的所有图像转换的组合
        transform = transforms.Compose([
            transforms.Resize(256),
            # 图像调整为256x256像素
            transforms.CenterCrop(224),
            # 将图像中心裁剪出来，大小为224x224像素
            transforms.ToTensor(),
            # 图像转换为tensor数据类型
            # 将图像的平均值和标准差设置为指定的值来正则化图像
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        return transform

    def load_data(self):
        # 对bert处理
        # segments = []
        attention_masks = []
        input_ids = []
        tokens = self.sentences
        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in tokens:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids.append(tkn['input_ids'].squeeze(0))
            attention_masks.append(tkn['attention_mask'].squeeze(0))

        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        # 对label处理
        y = [int(label) for label in self.y]
        y = torch.LongTensor(y)

        # 对img_label处理
        y_img = [int(label) for label in self.y_img]
        y_img = torch.LongTensor(y_img)

        # 对实体mids进行处理
        mid2index = self.make_dic()
        embedding = self.get_transe()  # torch.Size([86054151, 50])
        # print('transe size:',embedding.size())
        # count = 0
        vec_all_1, vec_all_2, sim_all = [], [], []

        #统计每个post的实体总数
        count_entity = 0
        for i, mid_post in enumerate(self.mids):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
                    count_entity += 1
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1.append(vec_post_1)
            vec_all_2.append(vec_post_2)
            sim_all.append(sim_post)
            # print(i)
        # return torch.LongTensor(vec_all)
        # print("vec_all_1",vec_all_1)
        # print("vec_all_2",vec_all_2)
        avg_count_entity = count_entity/len(self.mids)
        print('+++++++++++++++++++++++++++++++++++','\n')
        print('avg_count_entity:',avg_count_entity,'\n')
        print('+++++++++++++++++++++++++++++++++++','\n')
        # train\val\test
        fold0_test, fold0_val, fold0_train, \
        fold1_test, fold1_val, fold1_train, \
        fold2_test, fold2_val, fold2_train, \
        fold3_test, fold3_val, fold3_train, \
        fold4_test, fold4_val, fold4_train = split_data_5fold(len(y), y, self.config.train, self.config.val, \
                                                              self.config.test, shuffle=True)

        train_dict_0, val_dict_0, test_dict_0 = {}, {}, {}
        train_dict_1, val_dict_1, test_dict_1 = {}, {}, {}
        train_dict_2, val_dict_2, test_dict_2 = {}, {}, {}
        train_dict_3, val_dict_3, test_dict_3 = {}, {}, {}
        train_dict_4, val_dict_4, test_dict_4 = {}, {}, {}

        names_dict = {'input_ids': input_ids, 'attention_masks': attention_masks, 'image': self.images, \
                      'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img}
        for name in names_dict:
            train_dict_0[name] = [names_dict[name][i] for i in fold0_train]
            val_dict_0[name] = [names_dict[name][i] for i in fold0_val]
            test_dict_0[name] = [names_dict[name][i] for i in fold0_test]
        for name in names_dict:
            train_dict_1[name] = [names_dict[name][i] for i in fold1_train]
            val_dict_1[name] = [names_dict[name][i] for i in fold1_val]
            test_dict_1[name] = [names_dict[name][i] for i in fold1_test]
        for name in names_dict:
            train_dict_2[name] = [names_dict[name][i] for i in fold2_train]
            val_dict_2[name] = [names_dict[name][i] for i in fold2_val]
            test_dict_2[name] = [names_dict[name][i] for i in fold2_test]
        for name in names_dict:
            train_dict_3[name] = [names_dict[name][i] for i in fold3_train]
            val_dict_3[name] = [names_dict[name][i] for i in fold3_val]
            test_dict_3[name] = [names_dict[name][i] for i in fold3_test]
        for name in names_dict:
            train_dict_4[name] = [names_dict[name][i] for i in fold4_train]
            val_dict_4[name] = [names_dict[name][i] for i in fold4_val]
            test_dict_4[name] = [names_dict[name][i] for i in fold4_test]

        #保存npy格式数据，之后的实验直接加载.npy文件 直接存取字典
        # embedding = np.array(embedding)
        # # 保存npy文件
        # with open(clean_image_dir + '/image.npy', 'wb') as f:
        #     # root_index_final = np.array(root_index_final)
        #     np.save(f, embedding)
        # # print('finish')
        train_dict_list = [train_dict_0,train_dict_1,train_dict_2,train_dict_3,train_dict_4]
        val_dict_list = [val_dict_0, val_dict_1, val_dict_2, val_dict_3, val_dict_4]
        test_dict_list = [test_dict_0, test_dict_1, test_dict_2, test_dict_3, test_dict_4]

        save_split_path = '../data/{}/full_image/'.format(self.args.dataset)
        if not os.path.exists(save_split_path):
            os.mkdir(save_split_path)
        for i in range(5):
            np.save(save_split_path+'{}_split_train_{}'.format(self.args.dataset,i),train_dict_list[i])
            np.save(save_split_path + '{}_split_val_{}'.format(self.args.dataset, i), val_dict_list[i])
            np.save(save_split_path + '{}_split_test_{}'.format(self.args.dataset, i), test_dict_list[i])
        #npy文件读取
        # dict = {'a':{1,2,3},'b':{4,5,6}}
        # dict_load = np.load('xxx.npy',allow_pickle=True)
        # dict_load.item()
        # dict_load.item()['a']


        return train_dict_0, val_dict_0, test_dict_0, \
               train_dict_1, val_dict_1, test_dict_1, \
               train_dict_2, val_dict_2, test_dict_2, \
               train_dict_3, val_dict_3, test_dict_3, \
               train_dict_4, val_dict_4, test_dict_4
#weibo\pheme 伪实体补充random 变为seed 对平均每个post所包含实体和平均曼哈顿距离进行分析
class data_preprocess_ATT_bert_nfold_random_analysis():
    def __init__(self,sentences,images,mids,y,y_img,config,pathset,args): #sentence==token_ids
        #glove_path = '/home/sunmengzhu2019/kg_rumor/w2v/merge_sgns_bigram_char300.txt',
        #transE_path = '/home/sunmengzhu2019/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        # self.glove_path = pathset.path_glove
        self.sentences = sentences
        self.images = images
        self.mids = mids
        self.y = y
        self.y_img = y_img
        self.args = args

        self.sen_len = config.sen_len
        self.entity_len = config.entity_len

        self.transE_path = pathset.path_transe
        self.pathset = pathset
        self.config = config

        self.index2mid = []
        self.mid2index = {}
        self.index2word = []
        self.word2index = {}
        # self.embedding_matrix = []
        # self.embedding_glove = []

    def get_transe(self):
        vec = np.memmap(self.transE_path, dtype='float32', mode='r')
        # row = int(vec.shape[0]/50)
        self.embedding = vec.reshape((-1,50))
        self.embedding_entity_dim = 50 #50

        self.add_entity_embedding("<pad1>",2)
        self.add_entity_embedding("<pad2>",3)
        self.add_entity_embedding("<pad3>",4)
        self.add_entity_embedding("<pad4>",5)
        # #归一化
        # min_max_scaler = MinMaxScaler()
        # self.embedding_minmax = min_max_scaler.fit_transform(self.embedding)
        # return torch.tensor(self.embedding_minmax)
        #标准化
        mu = np.mean(self.embedding, axis=0)
        std = np.std(self.embedding, axis=0)
        self.embedding_mnstd = (self.embedding - mu) / std
        self.embedding_fin = torch.from_numpy(self.embedding_mnstd)
        return torch.tensor(self.embedding_fin)
        # return torch.tensor(self.embedding)

    def make_dic(self):
        dic_path = self.pathset.path_dic
        with open(dic_path,'r',encoding='utf-8')as t:
            text_all = t.readlines()
        for line in text_all[1:]:
            mid = line.split('\t')[0]
            idx = line.split('\t')[1].strip('\n')
            self.mid2index[mid] = int(idx)
        return self.mid2index

#TODO:random的问题
    def add_entity_embedding(self,entity,num):
        # vec_entity = torch.empty(1, 50)
        # nn.init.uniform_(vec_entity)

        #seed
        torch.manual_seed(num)
        vec_entity = torch.randn(1,50)
        self.mid2index[entity] = len(self.mid2index)
        # self.index2mid.append(entity)
        vec_entity = vec_entity.numpy()
        self.embedding = np.concatenate([self.embedding, vec_entity], 0)

    def top_dis_entity_cos(self,mid_post_index,embedding):
        # top_dis = 0
        top_dis = -1
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() > top_dis:
                        top_dis = cos_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def last_dis_entity_cos(self,mid_post_index,embedding):
        last_dis = 1
        last_vec1 = 0
        last_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() < last_dis:
                        last_dis = cos_sim.item()
                        last_vec1 = int(id1)
                        last_vec2 = int(id2)
                else:
                    continue
        print(last_dis)
        mid_post_list = []
        mid_post_list.append(last_vec1)
        mid_post_list.append(last_vec2)
        return mid_post_list

    def top_dis_entity_manhattan(self,mid_post_index, embedding):
        top_dis = 0
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    vec_sub = vec1-vec2
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    if manhattan_sim.item() > top_dis:
                        top_dis = manhattan_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def top_dis_5entity_cos(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    top_dis.append(cos_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[cos_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list

    def top_dis_5entity_manhattan(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    vec_sub = vec1 - vec2
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    top_dis.append(manhattan_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[manhattan_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        sim_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])
                sim_post_list.append(sim)

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list, sim_post_list

    def pad_eneity(self,mid,mid2index):
        if len(mid) < self.entity_len:
            pad_len = self.entity_len-len(mid)
            for j in range(pad_len):
                mid.append(mid2index["<pad{}>".format(j+1)])
        # assert len(mid) == self.entity_len
        # print(mid)
        return mid

    def img_trans(self):
        #定义一个变量transform，是对输入图像进行的所有图像转换的组合
        transform = transforms.Compose([
            transforms.Resize(256),
            #图像调整为256x256像素
            transforms.CenterCrop(224),
            #将图像中心裁剪出来，大小为224x224像素
            transforms.ToTensor(),
            #图像转换为tensor数据类型
            #将图像的平均值和标准差设置为指定的值来正则化图像
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )])
        return transform

    def load_data(self):
        #对bert处理
        # segments = []
        attention_masks = []
        input_ids = []
        tokens = self.sentences
        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in tokens:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids.append(tkn['input_ids'].squeeze(0))
            attention_masks.append(tkn['attention_mask'].squeeze(0))

        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        #对label处理
        y = [int(label) for label in self.y]
        y = torch.LongTensor(y)

        #对img_label处理
        y_img = [int(label) for label in self.y_img]
        y_img = torch.LongTensor(y_img)

        #对实体mids进行处理
        mid2index = self.make_dic()
        embedding = self.get_transe()  # torch.Size([86054151, 50])
        # print('transe size:',embedding.size())
        # count = 0
        vec_all_1, vec_all_2, sim_all = [],[],[]

        count_entity = 0
        for i, mid_post in enumerate(self.mids):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
                    count_entity += 1
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1.append(vec_post_1)
            vec_all_2.append(vec_post_2)
            sim_all.append(sim_post)
            # print(i)
        # return torch.LongTensor(vec_all)
        # print("vec_all_1",vec_all_1)
        # print("vec_all_2",vec_all_2)

        # avg_count_entity = count_entity/len(self.mids)
        avg_count_entity = count_entity / len(self.mids)
        print('+++++++++++++++++++++++++++++++++++', '\n')
        print('avg_count_entity:', avg_count_entity, '\n')
        print('+++++++++++++++++++++++++++++++++++', '\n')

        #输出csv文件，【label，sim_5】
        with open('t_test_entity_mahttanmax5_{}.csv'.format(self.args.dataset),'w',encoding='utf-8',newline='')as f:
            writer = csv.writer(f)
            for i in range(len(sim_all)):
                write_list = []
                label = y[i].item()
                write_list.append(label)
                print('SIM_ALL single len:',len(sim_all[i]),sim_all[i].size())
                #sim_all tensor torch.Size([1, 5])
                for j in range(5):
                    sim = sim_all[i][0][j].item()
                    write_list.append(sim)
                writer.writerow(write_list)

        #train\val\test
        fold0_test, fold0_val, fold0_train, \
        fold1_test, fold1_val, fold1_train, \
        fold2_test, fold2_val, fold2_train, \
        fold3_test, fold3_val, fold3_train, \
        fold4_test, fold4_val, fold4_train = split_data_5fold(len(y), y,self.config.train,self.config.val,\
                                                              self.config.test, shuffle=True)

        train_dict_0, val_dict_0, test_dict_0 = {}, {}, {}
        train_dict_1, val_dict_1, test_dict_1 = {}, {}, {}
        train_dict_2, val_dict_2, test_dict_2 = {}, {}, {}
        train_dict_3, val_dict_3, test_dict_3 = {}, {}, {}
        train_dict_4, val_dict_4, test_dict_4 = {}, {}, {}

        names_dict = {'input_ids':input_ids,'attention_masks':attention_masks,'image':self.images,\
                      'vec_1':vec_all_1, 'vec_2':vec_all_2, 'sim_list':sim_all,'y':y,'y_img':y_img}
        for name in names_dict:
            train_dict_0[name] = [names_dict[name][i] for i in fold0_train]
            val_dict_0[name] = [names_dict[name][i] for i in fold0_val]
            test_dict_0[name] = [names_dict[name][i] for i in fold0_test]
        for name in names_dict:
            train_dict_1[name] = [names_dict[name][i] for i in fold1_train]
            val_dict_1[name] = [names_dict[name][i] for i in fold1_val]
            test_dict_1[name] = [names_dict[name][i] for i in fold1_test]
        for name in names_dict:
            train_dict_2[name] = [names_dict[name][i] for i in fold2_train]
            val_dict_2[name] = [names_dict[name][i] for i in fold2_val]
            test_dict_2[name] = [names_dict[name][i] for i in fold2_test]
        for name in names_dict:
            train_dict_3[name] = [names_dict[name][i] for i in fold3_train]
            val_dict_3[name] = [names_dict[name][i] for i in fold3_val]
            test_dict_3[name] = [names_dict[name][i] for i in fold3_test]
        for name in names_dict:
            train_dict_4[name] = [names_dict[name][i] for i in fold4_train]
            val_dict_4[name] = [names_dict[name][i] for i in fold4_val]
            test_dict_4[name] = [names_dict[name][i] for i in fold4_test]



        return train_dict_0, val_dict_0, test_dict_0, \
               train_dict_1, val_dict_1, test_dict_1, \
               train_dict_2, val_dict_2, test_dict_2, \
               train_dict_3, val_dict_3, test_dict_3, \
               train_dict_4, val_dict_4, test_dict_4

#twitter 伪实体补充random 变为seed 对平均每个post所包含实体和平均曼哈顿距离进行分析
class data_preprocess_ATT_bert_random_analysis():
    def __init__(self,sentences_train,images_train,mids_train,y_train,y_img_train,\
                 sentences_test,images_test,mids_test,y_test,y_img_test,config,pathset,args): #sentence==token_ids
        #glove_path = '/home/sunmengzhu2019/kg_rumor/w2v/merge_sgns_bigram_char300.txt',
        #transE_path = '/home/sunmengzhu2019/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        # self.glove_path = pathset.path_glove
        self.sentences_train = sentences_train
        self.images_train = images_train
        self.mids_train = mids_train
        self.y_train = y_train
        self.y_img_train = y_img_train

        self.sentences_test = sentences_test
        self.images_test = images_test
        self.mids_test = mids_test
        self.y_test = y_test
        self.y_img_test = y_img_test

        self.sen_len = config.sen_len
        self.entity_len = config.entity_len

        self.transE_path = pathset.path_transe
        self.pathset = pathset
        self.config = config
        self.args = args

        self.index2mid = []
        self.mid2index = {}
        self.index2word = []
        self.word2index = {}
        # self.embedding_matrix = []
        # self.embedding_glove = []

    def get_transe(self):
        vec = np.memmap(self.transE_path, dtype='float32', mode='r')
        # row = int(vec.shape[0]/50)
        self.embedding = vec.reshape((-1,50))
        self.embedding_entity_dim = 50 #50

        self.add_entity_embedding("<pad1>",2)
        self.add_entity_embedding("<pad2>",3)
        self.add_entity_embedding("<pad3>",4)
        self.add_entity_embedding("<pad4>",5)
        # #归一化
        # min_max_scaler = MinMaxScaler()
        # self.embedding_minmax = min_max_scaler.fit_transform(self.embedding)
        #标准化
        mu = np.mean(self.embedding, axis=0)
        std = np.std(self.embedding, axis=0)
        self.embedding_mnstd = (self.embedding - mu) / std
        self.embedding_fin = torch.from_numpy(self.embedding_mnstd)
        return torch.tensor(self.embedding_fin)
        # return torch.tensor(self.embedding)

    def make_dic(self):
        dic_path = self.pathset.path_dic
        with open(dic_path,'r',encoding='utf-8')as t:
            text_all = t.readlines()
        for line in text_all[1:]:
            mid = line.split('\t')[0]
            idx = line.split('\t')[1].strip('\n')
            self.mid2index[mid] = int(idx)
        return self.mid2index

#TODO:RANDOM的问题
    def add_entity_embedding(self,entity,num):
        # vec_entity = torch.empty(1, 50)
        # nn.init.uniform_(vec_entity)
        #seed
        torch.manual_seed(num)
        vec_entity = torch.randn(1,50)
        self.mid2index[entity] = len(self.mid2index)
        # self.index2mid.append(entity)
        vec_entity = vec_entity.numpy()
        self.embedding = np.concatenate([self.embedding, vec_entity], 0)

    def top_dis_entity_cos(self,mid_post_index,embedding):
        # top_dis = 0
        top_dis = -1
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() > top_dis:
                        top_dis = cos_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def last_dis_entity_cos(self,mid_post_index,embedding):
        last_dis = 1
        last_vec1 = 0
        last_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1@vec2/(torch.sqrt(torch.sum(torch.pow(vec1,2)))*torch.sqrt(torch.sum(torch.pow(vec2,2))))
                    # print(dist)
                    if cos_sim.item() < last_dis:
                        last_dis = cos_sim.item()
                        last_vec1 = int(id1)
                        last_vec2 = int(id2)
                else:
                    continue
        print(last_dis)
        mid_post_list = []
        mid_post_list.append(last_vec1)
        mid_post_list.append(last_vec2)
        return mid_post_list

    def top_dis_entity_manhattan(self,mid_post_index, embedding):
        top_dis = 0
        top_vec1 = 0
        top_vec2 = 0
        # embedding = self.get_transe()
        for id1 in mid_post_index:
            for id2 in mid_post_index:
                if id2 != id1:
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    vec_sub = vec1-vec2
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    if manhattan_sim.item() > top_dis:
                        top_dis = manhattan_sim.item()
                        top_vec1 = int(id1)
                        top_vec2 = int(id2)
                else:
                    continue
        print(top_dis)
        mid_post_list = []
        mid_post_list.append(top_vec1)
        mid_post_list.append(top_vec2)
        return mid_post_list

    def top_dis_5entity_cos(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    cos_sim = vec1 @ vec2 / (
                                torch.sqrt(torch.sum(torch.pow(vec1, 2))) * torch.sqrt(torch.sum(torch.pow(vec2, 2))))
                    # print(dist)
                    top_dis.append(cos_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[cos_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list

    def top_dis_5entity_manhattan(self,mid_post_index,embedding,mid2index):
        top_dis = []
        top_vec1 = []
        top_vec2 = []
        dis_vec = {}

        # embedding = self.get_transe()
        # if len(mid_post_index) == 2:

        mid_post_index = self.pad_eneity(mid_post_index,mid2index) #len(mid_post_index)>=4
        valid_mid_post_index = [index for index in mid_post_index]
        for id1 in mid_post_index:
            valid_mid_post_index.remove(id1)
            # print("valid_mid_post_index:",valid_mid_post_index)
            if valid_mid_post_index:
                for id2 in valid_mid_post_index:
                    # if id2 != id1:
                    vec_list = []
                    vec1 = embedding[int(id1)]
                    vec2 = embedding[int(id2)]
                    # vec_sub = (vec1-vec2).unsqueeze(0)
                    # # print(vec_sub.size())
                    # dist = torch.sqrt(torch.mm(vec_sub,vec_sub.transpose(0,1)))
                    vec_sub = vec1 - vec2
                    manhattan_sim = vec_sub.norm(1)
                    # print(dist)
                    top_dis.append(manhattan_sim)
                    vec_list.append(int(id1))
                    vec_list.append(int(id2))
                    dis_vec[manhattan_sim] = vec_list
        # top_dis.sort(reverse=True)
        count_sim = 0
        mid_post_list = []
        sim_post_list = []
        for sim in sorted(dis_vec,reverse=True):   #sorted 由小到大
            if count_sim >= 5:
                break
            else:
                count_sim += 1
                # print('top sim:',sim,'vec_list:',dis_vec[sim])
                mid_post_list.append(dis_vec[sim])
                sim_post_list.append(sim)

        # print(top_dis)
        # mid_post_list = []
        # for i in range(len(top_dis)):
        #     mid_post_list.append(dis_vec[top_dis[i]])

        # mid_post_list = list(set(mid_post_list))
        return mid_post_list, sim_post_list

    def pad_eneity(self,mid,mid2index):
        if len(mid) < self.entity_len:
            pad_len = self.entity_len-len(mid)
            for j in range(pad_len):
                mid.append(mid2index["<pad{}>".format(j+1)])
        # assert len(mid) == self.entity_len
        # print(mid)
        return mid

    def img_trans(self):
        #定义一个变量transform，是对输入图像进行的所有图像转换的组合
        transform = transforms.Compose([
            transforms.Resize(256),
            #图像调整为256x256像素
            transforms.CenterCrop(224),
            #将图像中心裁剪出来，大小为224x224像素
            transforms.ToTensor(),
            #图像转换为tensor数据类型
            #将图像的平均值和标准差设置为指定的值来正则化图像
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )])
        return transform

    def load_data(self):
        #对bert处理
        # segments = []
        attention_masks_train = []
        input_ids_train = []
        tokens_train = self.sentences_train

        attention_masks_test = []
        input_ids_test = []
        tokens_test = self.sentences_test

        # print(tkn['input_ids'],type(tkn['input_ids']))
        # print(type(tokens['input_ids']),tokens['input_ids'])
        for tkn in tokens_train:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids_train.append(tkn['input_ids'].squeeze(0))
            attention_masks_train.append(tkn['attention_mask'].squeeze(0))

        for tkn in tokens_test:
            # print(tkn['input_ids'],type(tkn['input_ids']))
            input_ids_test.append(tkn['input_ids'].squeeze(0))
            attention_masks_test.append(tkn['attention_mask'].squeeze(0))
        # input_ids_tensor = torch.tensor(input_ids)
        # segments_tensors = torch.tensor(segments)
        # attention_masks_tensors = torch.tensor(attention_masks)

        #对label处理
        y_train = [int(label) for label in self.y_train]
        y_train = torch.LongTensor(y_train)

        y_test = [int(label) for label in self.y_test]
        y_test = torch.LongTensor(y_test)

        #对img_label处理
        y_img_train = [int(label) for label in self.y_img_train]
        y_img_train = torch.LongTensor(y_img_train)

        y_img_test = [int(label) for label in self.y_img_test]
        y_img_test = torch.LongTensor(y_img_test)

        #对实体mids进行处理
        mid2index = self.make_dic()
        embedding = self.get_transe()  # torch.Size([86054151, 50])
        # print('transe size:',embedding.size())
        # count = 0
        vec_all_1_train, vec_all_2_train, sim_all_train = [],[],[]

        count_entity = 0
        for i, mid_post in enumerate(self.mids_train):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
                    count_entity += 1
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1_train.append(vec_post_1)
            vec_all_2_train.append(vec_post_2)
            sim_all_train.append(sim_post)
            # print(i)
        # return torch.LongTensor(vec_all)
        # print("vec_all_1",vec_all_1)
        # print("vec_all_2",vec_all_2)
        vec_all_1_test, vec_all_2_test, sim_all_test = [], [], []

        for i, mid_post in enumerate(self.mids_test):
            mid_post_index = []
            for mid in mid_post:
                # print(mid)
                if mid in mid2index:
                    mid_post_index.append(mid2index[mid])
                    count_entity += 1
            # print(mid_post_index)
            # print("len mid_post:",len(mid_post))
            # print("len mid_post_index:", len(mid_post_index))
            mid_post_index, sim_post_list = self.top_dis_5entity_manhattan(mid_post_index, embedding, mid2index)
            # print(embedding.shape)
            # print(mid_post_index)
            # print(mid_post_index[1])
            sim_post = torch.tensor(sim_post_list)
            sim_post = sim_post.unsqueeze(0)
            vec_post_1_1 = embedding[mid_post_index[0][0], :].unsqueeze(0)
            vec_post_1_2 = embedding[mid_post_index[0][1], :].unsqueeze(0)
            vec_post_2_1 = embedding[mid_post_index[1][0], :].unsqueeze(0)
            vec_post_2_2 = embedding[mid_post_index[1][1], :].unsqueeze(0)
            vec_post_3_1 = embedding[mid_post_index[2][0], :].unsqueeze(0)
            vec_post_3_2 = embedding[mid_post_index[2][1], :].unsqueeze(0)
            vec_post_4_1 = embedding[mid_post_index[3][0], :].unsqueeze(0)
            vec_post_4_2 = embedding[mid_post_index[3][1], :].unsqueeze(0)
            vec_post_5_1 = embedding[mid_post_index[4][0], :].unsqueeze(0)
            vec_post_5_2 = embedding[mid_post_index[4][1], :].unsqueeze(0)
            # print("vec_post_1",vec_post_1)
            # print("vec_post_2",vec_post_2)
            vec_post_1 = torch.cat((vec_post_1_1, vec_post_2_1, vec_post_3_1, vec_post_4_1, vec_post_5_1),
                                   axis=0)
            vec_post_2 = torch.cat((vec_post_1_2, vec_post_2_2, vec_post_3_2, vec_post_4_2, vec_post_5_2),
                                   axis=0)
            sim_post = sim_post

            # print('data_process:',vec_post_1.size(),vec_post_2.size(),sim_post.size())
            # if i == 0:
            #     vec_all_1 = vec_post_1
            #     vec_all_2 = vec_post_2
            #     sim_all = sim_post.unsqueeze(0)
            # else:
            #     vec_all_1 = torch.cat((vec_all_1, vec_post_1), axis=0)
            #     vec_all_2 = torch.cat((vec_all_2, vec_post_2), axis=0)
            #     sim_all = torch.cat((sim_all, sim_post.unsqueeze(0)), axis=0)#均为三维矩阵
            vec_all_1_test.append(vec_post_1)
            vec_all_2_test.append(vec_post_2)
            sim_all_test.append(sim_post)

        avg_count_entity = count_entity / (len(self.mids_train)+len(self.mids_test))
        print('+++++++++++++++++++++++++++++++++++', '\n')
        print('avg_count_entity:', avg_count_entity, '\n')
        print('+++++++++++++++++++++++++++++++++++', '\n')
        #train\val\test
        with open('t_test_entity_mahttanmax5_{}.csv'.format(self.args.dataset), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(sim_all_train)):
                write_list = []
                label = y_train[i].item()
                write_list.append(label)
                for j in range(5):
                    sim = sim_all_train[i][0][j].item()
                    write_list.append(sim)
                writer.writerow(write_list)
            for i in range(len(sim_all_test)):
                write_list = []
                label = y_test[i].item()
                write_list.append(label)
                for j in range(5):
                    sim = sim_all_test[i][0][j].item()
                    write_list.append(sim)
                writer.writerow(write_list)

        fold_val_0, fold_test_0 = split_data(len(y_test), y_test,self.config.val_en,\
                                                              self.config.test_en, shuffle=True)
        fold_val_1, fold_test_1 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)
        fold_val_2, fold_test_2 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)
        fold_val_3, fold_test_3 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)
        fold_val_4, fold_test_4 = split_data(len(y_test), y_test, self.config.val_en, \
                                         self.config.test_en, shuffle=True)


        train_dict_0, val_dict_0, test_dict_0 = {}, {}, {}
        train_dict_1, val_dict_1, test_dict_1 = {}, {}, {}
        train_dict_2, val_dict_2, test_dict_2 = {}, {}, {}
        train_dict_3, val_dict_3, test_dict_3 = {}, {}, {}
        train_dict_4, val_dict_4, test_dict_4 = {}, {}, {}

        names_dict_test = {'input_ids':input_ids_test,'attention_masks':attention_masks_test,'image':self.images_test,\
                      'vec_1':vec_all_1_test, 'vec_2':vec_all_2_test, 'sim_list':sim_all_test,'y':y_test,'y_img':y_img_test}
        for name in names_dict_test:
            # train_dict_0[name] = [names_dict[name][i] for i in fold0_train]
            val_dict_0[name] = [names_dict_test[name][i] for i in fold_val_0]
            val_dict_1[name] = [names_dict_test[name][i] for i in fold_val_1]
            val_dict_2[name] = [names_dict_test[name][i] for i in fold_val_2]
            val_dict_3[name] = [names_dict_test[name][i] for i in fold_val_3]
            val_dict_4[name] = [names_dict_test[name][i] for i in fold_val_4]

            test_dict_0[name] = [names_dict_test[name][i] for i in fold_test_0]
            test_dict_1[name] = [names_dict_test[name][i] for i in fold_test_1]
            test_dict_2[name] = [names_dict_test[name][i] for i in fold_test_2]
            test_dict_3[name] = [names_dict_test[name][i] for i in fold_test_3]
            test_dict_4[name] = [names_dict_test[name][i] for i in fold_test_4]

        names_dict_train = {'input_ids': input_ids_train, 'attention_masks': attention_masks_train,
                            'image': self.images_train, \
                            'vec_1': vec_all_1_train, 'vec_2': vec_all_2_train, 'sim_list': sim_all_train, 'y': y_train,
                            'y_img': y_img_train}
        for name in names_dict_train:
            train_dict_0[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_1[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_2[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_3[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            train_dict_4[name] = [names_dict_train[name][i] for i in range(len(y_train))]
            # val_dict[name] = [names_dict_train[name][i] for i in fold_val]
            # test_dict[name] = [names_dict_train[name][i] for i in fold_test]



        return train_dict_0, val_dict_0, test_dict_0, \
               train_dict_1, val_dict_1, test_dict_1, \
               train_dict_2, val_dict_2, test_dict_2, \
               train_dict_3, val_dict_3, test_dict_3, \
               train_dict_4, val_dict_4, test_dict_4


# masking 根据train_mask_per test_mask_per确定mask的比例
#
class data_preprocess_ATT_bert_masking():
    def __init__(self, train_dict, val_dict, test_dict, config, pathset, args):  # sentence==token_ids
        # glove_path = '/home/sunmengzhu2019/kg_rumor/w2v/merge_sgns_bigram_char300.txt',
        # transE_path = '/home/sunmengzhu2019/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        # self.glove_path = pathset.path_glove
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.test_dict = test_dict

        self.pathset = pathset
        self.config = config

        self.args = args

    def img_trans(self):
        # 定义一个变量transform，是对输入图像进行的所有图像转换的组合
        transform = transforms.Compose([
            transforms.Resize(256),
            # 图像调整为256x256像素
            transforms.CenterCrop(224),
            # 将图像中心裁剪出来，大小为224x224像素
            transforms.ToTensor(),
            # 图像转换为tensor数据类型
            # 将图像的平均值和标准差设置为指定的值来正则化图像
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        return transform

    def load_data(self):
        # names_dict = {'input_ids': input_ids, 'attention_masks': attention_masks, 'image': self.images, \
        #               'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img}

        # train_input_ids = self.train_dict['input_ids']
        # train_atteneion_masks = self.train_dict['attention_masks']
        #训练集
        train_image = self.train_dict['image']


        # train_vec_1 = self.train_dict['vec_1']
        # train_vec_2 = self.train_dict['vec_2']
        # train_sim_list = self.train_dict['sim_list']
        # train_y = self.train_dict['y']
        train_y_img = self.train_dict['y_img']

        # 查看数据类型
        print('train_image type:', type(train_image)) #list
        print('train_y_img type:', type(train_y_img))# list

        print('train_image type:', train_image[0:10], type(train_image[0]))  # list
        print('train_y_img type:', train_y_img[0:10], type(train_y_img[0]), train_y_img[0].size())  # list
        #train_image type: ['002648', '001655', '002017', '001435', '002706', '003563', '000059', '001222', '001914', '003530'] <class 'numpy.str_'>
        # train_y_img type: [tensor(1), tensor(1), tensor(1), tensor(1), tensor(1), tensor(1), tensor(1), tensor(1), tensor(1), tensor(1)] <class 'torch.Tensor'> torch.Size([])

        idx_len_mask_arr_train = data_masking(train_image,train_y_img,self.args.train_mask_per)
        print('idx_len_mask_arr_train:', idx_len_mask_arr_train[0:10]) #idx_len_mask_arr_train: [51, 757, 631, 14, 1317, 310, 1137, 903, 359, 725]
        # 每次一样的  [51, 757, 631, 14, 1317, 310, 1137, 903, 359, 725]


        for idx in idx_len_mask_arr_train:
            self.train_dict['image'][idx] = 'none'
            self.train_dict['y_img'][idx] = torch.tensor(0)

        #验证集
        val_image = self.val_dict['image']
        val_y_img = self.val_dict['y_img']

        idx_len_mask_arr_val = data_masking(val_image, val_y_img, self.args.test_mask_per)
        for idx in idx_len_mask_arr_val:
            self.val_dict['image'][idx] = 'none'
            self.val_dict['y_img'][idx] = torch.tensor(0)

        #测试集
        test_image = self.test_dict['image']
        test_y_img = self.test_dict['y_img']

        idx_len_mask_arr_test = data_masking(test_image, test_y_img, self.args.test_mask_per)
        for idx in idx_len_mask_arr_test:
            self.test_dict['image'][idx] = 'none'
            self.test_dict['y_img'][idx] = torch.tensor(0)



        return self.train_dict, self.val_dict, self.test_dict

def data_masking(image,y_img,mask_per):
    idx_len = len(image)
    mask_len = int(idx_len*mask_per)
    idx_len_arr = [i for i in range(idx_len)]
    #每次运行固定shuffle次序，对随机序列的前mask_per*100%的将image置为none，y_img置为0（无图）
    random.seed(0)
    random.shuffle(idx_len_arr)
    idx_len_mask_arr = idx_len_arr[:mask_len]

    return idx_len_mask_arr




