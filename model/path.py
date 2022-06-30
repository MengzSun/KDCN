import os
import sys

class path_set_BERT():
    def __init__(self,dataset):
        self.path_data_dir = os.path.join('/home/sunmengzhu2019/tkde_kg_rumor/','data/{}'.format(dataset))
        #text\image\label
        self.path_white_img = os.path.join('../data/white.jpg')
        if dataset == 'twitter':
            self.path_txt_data_train = os.path.join(self.path_data_dir,'en_single_wimg_went_train.csv')
            self.path_txt_data_test = os.path.join(self.path_data_dir,'en_single_wimg_went_test.csv')
            self.path_img_data = '/home/sunmengzhu2019/kg_rumor/raw_data/en_image/en_image_all/'
        else:
            self.path_txt_data = os.path.join(self.path_data_dir,'{}_single_wimg_went.csv'.format(dataset))
            if dataset == 'pheme':
                self.path_img_data = '/home/sunmengzhu2019/kg_rumor/raw_data_pheme/pheme_big/images/'
            elif dataset == 'weibo':
                self.path_img_data = '/home/sunmengzhu2019/kg_rumor/raw_data_zh/image_rename/zh_image_all/'


        #trained_model
        self.path_saved_model = 'model_saved/'

        #BERT_PATH
        if dataset == 'weibo':
            self.path_bert = '../bert-base-chinese/'
            self.VOCAB = 'vocab.txt'
        else:
            self.path_bert = '../bert-base-uncased/'
            self.VOCAB = 'vocab.txt'
        #TransE path
        self.path_transe = '/home/sunmengzhu2019/Freebase/embeddings/dimension_50/transe/entity2vec.bin'
        self.path_dic = '/home/sunmengzhu2019/Freebase/knowledge graphs/entity2id.txt'



