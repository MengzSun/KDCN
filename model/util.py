import csv
import os

import numpy as np
import torch
import torch.nn as nn
import re
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from transformers import BertModel, BertTokenizer, BertConfig

class StringProcess(object):
    def __init__(self):
        self.other_char = re.compile(r"[^A-Za-z0-9(),!?\'\`]", flags=0)
        self.num = re.compile(r"[+-]?\d+\.?\d*", flags=0)
        # self.url = re.compile(r"[a-z]*[:.]+\S+|\n|\s+", flags=0)
        self.url = re.compile(
                r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", flags=0)
        self.stop_words = None
        self.nlp = None

    def clean_str(self, string):
        # string = re.sub(self.other_char, " ", string)
        # string = re.sub(r"\'s", " \'s", string)
        # string = re.sub(r"\'ve", " \'ve", string)
        # string = re.sub(r"n\'t", " n\'t", string)
        # string = re.sub(r"\'re", " \'re", string)
        # string = re.sub(r"\'d", " \'d", string)
        # string = re.sub(r"\'ll", " \'ll", string)
        # string = re.sub(r",", " , ", string)
        # string = re.sub(r"!", " ! ", string)
        # string = re.sub(r"\(", " \( ", string)
        # string = re.sub(r"\)", " \) ", string)
        # string = re.sub(r"\?", " \? ", string)
        # string = re.sub(r"\s{2,}", " ", string)
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        string = string.split('http')[0]
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr,' ',string)
        string = re.sub(r4,' ',string)
        string = string.strip().lower()
        string = self.remove_stopword(string)

        return string

    def clean_str_zh(self, string):
        # string = re.sub(self.other_char, " ", string)
        # string = re.sub(r"\'s", " \'s", string)
        # string = re.sub(r"\'ve", " \'ve", string)
        # string = re.sub(r"n\'t", " n\'t", string)
        # string = re.sub(r"\'re", " \'re", string)
        # string = re.sub(r"\'d", " \'d", string)
        # string = re.sub(r"\'ll", " \'ll", string)
        # string = re.sub(r",", " , ", string)
        # string = re.sub(r"!", " ! ", string)
        # string = re.sub(r"\(", " \( ", string)
        # string = re.sub(r"\)", " \) ", string)
        # string = re.sub(r"\?", " \? ", string)
        # string = re.sub(r"\s{2,}", " ", string)
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr, ' ', string)
        string = re.sub(r4, ' ', string)
        string = string.strip()
        string = self.remove_stopword_zh(string)
        return string

    def clean_str_BERT(self,string):
        r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
        r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
        r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
        r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
        string = string.split('http')[0]
        cleanr = re.compile('<.*?>')
        string = re.sub(cleanr, ' ', string)
        # string = re.sub(r1, ' ', string)
        # string = re.sub(r2, ' ', string)
        # string = re.sub(r3, ' ', string)
        string = re.sub(r4, ' ', string)
        return string

    def norm_str(self, string):
        string = re.sub(self.other_char, " ", string)

        if self.nlp is None:
            from spacy.lang.en import English
            self.nlp = English()

        new_doc = list()
        doc = self.nlp(string)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_digit:
                token = "[num]"
            else:
                token = token.text

            new_doc.append(token)

        return " ".join(new_doc).lower()

    def lean_str_sst(self, string):
        """
            Tokenization/string cleaning for the SST yelp_dataset
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        string = re.sub(self.other_char, " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def remove_stopword(self, string):
        if self.stop_words is None:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = string.split()

        new_string = list()
        for word in string:
            if word in self.stop_words:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def remove_stopword_zh(self, string):
        stopwords = []
        with open('../data/weibo/stop_words.txt', 'r', encoding='utf-8')as f:
            txt = f.readlines()
        for line in txt:
            # print(line.strip('\n'))
            stopwords.append(line.strip('\n'))

        # if self.stop_words is None:
        #     from nltk.corpus import stopwords
        #     self.stop_words = set(stopwords.words('english'))

        if type(string) is str:
            string = jieba.cut(string)

        new_string = list()
        for word in string:
            if word in stopwords:
                continue
            new_string.append(word)

        return " ".join(new_string)

    def replace_num(self, string):
        result = re.sub(self.num, '<num>', string)
        return result

    def replace_urls(self, string):
        result = re.sub(self.url, '<url>', string)
        result = ' '.join(re.split(' +|\n+', result)).strip()
        return result

def pre_training_pheme(pathset, config):
    text_id = []  # 592595287815757825\t\t 这种格式的
    tweet = []
    image_id = []
    label = []  # fake\n 这种格式的
    img_label = []
    mids = []
    with open(pathset.path_txt_data, 'r', encoding='utf-8')as f:
        reader = csv.reader(f)
        for line in reader:
            # print(line[0])
            text_id.append(line[0].strip('\t').strip('\ufeff').strip('"').strip('\t'))
            tweet.append(line[1].strip('\t'))
            image_id.append(line[2].strip('\t').strip('\ufeff').strip('"').strip('\t'))
            label.append(int(line[3].strip('\t')))
            mids.append(line[4].strip('\t'))
    # print(len(text_id),len(tweet),len(image_id),len(label)) 总数据16417个
    # print(label)

    # r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
    # r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
    # r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
    # r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    #
    # for i in range(len(tweet)):
    #     sentence = tweet[i].split('http')[0]
    #     cleanr = re.compile('<.*?>')
    #     sentence = re.sub(cleanr, ' ', sentence)
    #     sentence = re.sub(r4, '', sentence)
    #     tweet[i] = sentence
    #     # print(sentence)
    # # print(tweet)
    UNCASED = pathset.path_bert
    VOCAB = pathset.VOCAB
    tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
    tokens_ids = []
    # for tw in tweet:
    #     tw = "[CLS] " + tw + " [SEP]"
    #     tkn = tokenizer.tokenize(tw)
    #     tkn_id = tokenizer.convert_tokens_to_ids(tkn)
    #     tokens_ids.append(tkn_id)
    string_process = StringProcess()
    for i in range(len(tweet)):
        sentence = string_process.clean_str_BERT(tweet[i])
        tokenizer_encoding = tokenizer(sentence, return_tensors='pt', padding='max_length',\
                                       truncation=True, max_length=config.sen_len)
        tokens_ids.append(tokenizer_encoding)

    for idx in image_id:
        if idx == 'none':
            img_label.append(0)
        else:
            img_label.append(1)

    mids_all = []
    for i in range(len(text_id)):
        mid = []
        # print(i)
        # print(image_id_train[i])
        # if image_id_train[i] in img2mid:
        #     img_mid = img2mid[image_id_train[i]]
        # mid = mids_txt[i] + img_mid
        for en in mids[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
            # mid.append(en)

        mid = list(set(mid))
        mids_all.append(mid)

    # print('util:',tokens_ids)
    # {'input_ids': tensor([[  101,  2292,  2149, 28887,  1996,  1000,  3994,  7668, 15126,  1000,
    #          23713,  1025, 11839,  2005,  1996,  3647,  2709,  1997, 19323,  2292,
    #           1055, 15908,  2000,  4652,  2107,  4490,  1997,  4808,  2408,  1996,
    #           2088,   102,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #          1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0]])}
    # X_txt = np.array(tokens_ids)
    X_txt = tokens_ids
    X_img = np.array(image_id)

    X_kg = np.array(mids_all)

    # X = np.array(tokens_ids)
    y = np.array(label)
    y_img = np.array(img_label)

    # train_txt_x_temp,test_txt_x, train_img_x_temp,test_img_x, train_kg_x_temp,test_kg_x, train_y_temp,test_y = train_test_split(
    #     X_txt,X_img,X_kg,y, test_size=0.3,
    #     stratify=y,
    #     shuffle=True,
    #     random_state=1
    # )
    # ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    # train_idx, val_idx = next(ss.split(train_txt_x_temp, train_y_temp))
    # train_txt_x,train_img_x,train_kg_x,train_y = train_txt_x_temp[train_idx],train_img_x_temp[train_idx],train_kg_x_temp[train_idx],train_y_temp[train_idx]
    # val_txt_x, val_img_x, val_kg_x, val_y = train_txt_x_temp[val_idx], train_img_x_temp[val_idx], \
    #                                                 train_kg_x_temp[val_idx], train_y_temp[val_idx]

    # return text_id, tweet, image_id, label, mids_all, \
    #        train_txt_x, train_img_x, train_kg_x, train_y, \
    #        val_txt_x, val_img_x, val_kg_x, val_y, \
    #        test_txt_x, test_img_x, test_kg_x, test_y
    return X_txt, X_img, X_kg, y, y_img

def pre_training_en(pathset,config):
    text_id_train = []  # 592595287815757825\t\t 这种格式的
    tweet_train = []
    image_id_train = []
    label_train = []  # fake\n 这种格式的
    img_label_train = []
    mid_train = []
    with open(pathset.path_txt_data_train, 'r', encoding='utf-8')as f:
        reader = csv.reader(f)
        for line in reader:
            text_id_train.append(line[0].strip('\t').strip('\ufeff').strip('"').strip('\t'))
            tweet_train.append(line[1].strip('\t'))
            image_id_train.append(line[2].strip('\t'))
            label_train.append(int(line[3].strip('\t')))
            # mid = [m.strip(' ').strip('\'') for m in line[4].strip('\t').strip('[').strip(']').split(',')]
            mid_train.append(line[4].strip('\t'))
    # print(len(text_id),len(tweet),len(image_id),len(label)) 总数据16417个
    # print(label)
    text_id_test = []  # 592595287815757825\t\t 这种格式的
    tweet_test = []
    image_id_test = []
    label_test = []  # fake\n 这种格式的
    img_label_test = []
    mid_test = []
    with open(pathset.path_txt_data_test, 'r', encoding='utf-8')as f:
        reader = csv.reader(f)
        for line in reader:
            text_id_test.append(line[0].strip('\t').strip('\ufeff').strip('"').strip('\t'))
            tweet_test.append(line[1].strip('\t'))
            image_id_test.append(line[2].strip('\t'))
            label_test.append(int(line[3].strip('\t')))
            # mid = [m.strip(' ').strip('\'') for m in line[4].strip('\t').strip('[').strip(']').split(',')]
            mid_test.append(line[4].strip('\t'))

    UNCASED = pathset.path_bert
    VOCAB = pathset.VOCAB
    tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
    tokens_ids_train = []
    tokens_ids_test = []
    string_process = StringProcess()
    for i in range(len(tweet_train)):
        sentence = string_process.clean_str_BERT(tweet_train[i])
        tokenizer_encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', \
                                       truncation=True, max_length=config.sen_len)
        tokens_ids_train.append(tokenizer_encoding)

    for i in range(len(tweet_test)):
        sentence = string_process.clean_str_BERT(tweet_test[i])
        tokenizer_encoding = tokenizer(sentence, return_tensors='pt', padding='max_length',\
                                       truncation=True, max_length=config.sen_len)
        tokens_ids_test.append(tokenizer_encoding)

    for idx in image_id_train:
        if idx == 'none':
            img_label_train.append(0)
        else:
            img_label_train.append(1)
    for idx in image_id_test:
        if idx == 'none':
            img_label_test.append(0)
        else:
            img_label_test.append(1)

    mids_train_all = []
    for i in range(len(text_id_train)):
        mid = []
        # print(i)
        # print(image_id_train[i])
        # if image_id_train[i] in img2mid:
        #     img_mid = img2mid[image_id_train[i]]
        # mid = mids_txt[i] + img_mid
        for en in mid_train[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
            # mid.append(en)
        mid = list(set(mid))
        mids_train_all.append(mid)

    mids_test_all = []
    for i in range(len(text_id_test)):
        mid = []
        # print(i)
        # print(image_id_train[i])
        # if image_id_train[i] in img2mid:
        #     img_mid = img2mid[image_id_train[i]]
        # mid = mids_txt[i] + img_mid
        for en in mid_test[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
            # mid.append(en)
        mid = list(set(mid))
        mids_test_all.append(mid)

    tokens_ids = tokens_ids_train + tokens_ids_test
    # max_len = max([len(single) for single in tokens_ids])
    # 划分数据集 以及读取label_index

    X_txt_train = tokens_ids_train
    X_txt_test = tokens_ids_test
    X_img_train = np.array(image_id_train)
    X_img_test = np.array(image_id_test)
    X_kg_train = np.array(mids_train_all)
    X_kg_test = np.array(mids_test_all)
    # X = np.array(tokens_ids_train)
    y_train = np.array(label_train)
    y_test = np.array(label_test)
    y_img_train = np.array(img_label_train)
    y_img_test = np.array(img_label_test)
    # print(X.size)
    # print(y_train.size)


    return X_txt_train, X_img_train, X_kg_train, y_train, y_img_train, \
           X_txt_test, X_img_test, X_kg_test, y_test, y_img_test

def pre_training_zh(pathset,config):
    text_id = []  # 592595287815757825\t\t 这种格式的
    tweet = []
    image_id = []
    label = []  # fake\n 这种格式的
    img_label = []
    mids = []
    with open(pathset.path_txt_data, 'r', encoding='utf-8')as f:
        reader = csv.reader(f)
        for line in reader:
            # print(line[0])
            text_id.append(line[0].strip('\t').strip('\ufeff').strip('"').strip('\t'))
            tweet.append(line[1].strip('\t'))
            image_id.append(line[2].strip('\t').strip('\ufeff').strip('"').strip('\t'))
            label.append(int(line[3].strip('\n')))
            mids.append(line[4].strip('\n'))
    # print(label)
    # print(type(label[0]))
    UNCASED = pathset.path_bert
    VOCAB = pathset.VOCAB
    tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
    tokens_ids = []
    # for tw in tweet:
    #     tw = "[CLS] " + tw + " [SEP]"
    #     tkn = tokenizer.tokenize(tw)
    #     tkn_id = tokenizer.convert_tokens_to_ids(tkn)
    #     tokens_ids.append(tkn_id)
    string_process = StringProcess()
    for i in range(len(tweet)):
        sentence = string_process.clean_str_BERT(tweet[i])
        tokenizer_encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', \
                                       truncation=True, max_length=config.sen_len)
        tokens_ids.append(tokenizer_encoding)
    # mids_all = []
    # for mid in mids:
    #     mids_lst = [m.strip(' ').strip('\'') for m in mid.strip('[').strip(']').split(',')]
    #     # print(mids_lst)
    #     mids_all.append(mids_lst)
    for idx in image_id:
        if idx == 'none':
            img_label.append(0)
        else:
            img_label.append(1)


    mids_all = []
    for i in range(len(text_id)):
        mid = []
        for en in mids[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
        mid = list(set(mid))
        mids_all.append(mid)
    # 划分数据集 以及读取label_index

    # X_txt = np.array(tokens_ids)
    X_txt = tokens_ids
    X_img = np.array(image_id)
    X_kg = np.array(mids_all)
    # X = np.array(tokens_ids)
    y = np.array(label)
    y_img = np.array(img_label)

    return X_txt, X_img, X_kg, y, y_img

#读取图片模态不缺失的数据集
def pre_training_pheme_full_image(pathset, config):
    text_id = []  # 592595287815757825\t\t 这种格式的
    tweet = []
    image_id = []
    label = []  # fake\n 这种格式的
    img_label = []
    mids = []
    with open(pathset.path_txt_data, 'r', encoding='utf-8')as f:
        reader = csv.reader(f)
        for line in reader:
            # print(line[0])
            #判断是否有图
            img = line[2].strip('\t').strip('\ufeff').strip('"').strip('\t')
            if img != 'none':
                text_id.append(line[0].strip('\t').strip('\ufeff').strip('"').strip('\t'))
                tweet.append(line[1].strip('\t'))
                image_id.append(line[2].strip('\t').strip('\ufeff').strip('"').strip('\t'))
                label.append(int(line[3].strip('\t')))
                mids.append(line[4].strip('\t'))
    # print(len(text_id),len(tweet),len(image_id),len(label)) 总数据16417个
    # print(label)

    # r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
    # r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
    # r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
    # r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    #
    # for i in range(len(tweet)):
    #     sentence = tweet[i].split('http')[0]
    #     cleanr = re.compile('<.*?>')
    #     sentence = re.sub(cleanr, ' ', sentence)
    #     sentence = re.sub(r4, '', sentence)
    #     tweet[i] = sentence
    #     # print(sentence)
    # # print(tweet)
    UNCASED = pathset.path_bert
    VOCAB = pathset.VOCAB
    tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
    tokens_ids = []
    # for tw in tweet:
    #     tw = "[CLS] " + tw + " [SEP]"
    #     tkn = tokenizer.tokenize(tw)
    #     tkn_id = tokenizer.convert_tokens_to_ids(tkn)
    #     tokens_ids.append(tkn_id)
    string_process = StringProcess()
    for i in range(len(tweet)):
        sentence = string_process.clean_str_BERT(tweet[i])
        tokenizer_encoding = tokenizer(sentence, return_tensors='pt', padding='max_length',\
                                       truncation=True, max_length=config.sen_len)
        tokens_ids.append(tokenizer_encoding)

    for idx in image_id:
        if idx == 'none':
            img_label.append(0)
        else:
            img_label.append(1)

    #加个断言
    for lbl in img_label:
        assert lbl == 1

    mids_all = []
    for i in range(len(text_id)):
        mid = []
        # print(i)
        # print(image_id_train[i])
        # if image_id_train[i] in img2mid:
        #     img_mid = img2mid[image_id_train[i]]
        # mid = mids_txt[i] + img_mid
        for en in mids[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
            # mid.append(en)

        mid = list(set(mid))
        mids_all.append(mid)

    # print('util:',tokens_ids)
    # {'input_ids': tensor([[  101,  2292,  2149, 28887,  1996,  1000,  3994,  7668, 15126,  1000,
    #          23713,  1025, 11839,  2005,  1996,  3647,  2709,  1997, 19323,  2292,
    #           1055, 15908,  2000,  4652,  2107,  4490,  1997,  4808,  2408,  1996,
    #           2088,   102,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #              0,     0,     0,     0,     0,     0,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #          1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0]])}
    # X_txt = np.array(tokens_ids)
    X_txt = tokens_ids
    X_img = np.array(image_id)

    X_kg = np.array(mids_all)

    # X = np.array(tokens_ids)
    y = np.array(label)
    y_img = np.array(img_label)

    # train_txt_x_temp,test_txt_x, train_img_x_temp,test_img_x, train_kg_x_temp,test_kg_x, train_y_temp,test_y = train_test_split(
    #     X_txt,X_img,X_kg,y, test_size=0.3,
    #     stratify=y,
    #     shuffle=True,
    #     random_state=1
    # )
    # ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    # train_idx, val_idx = next(ss.split(train_txt_x_temp, train_y_temp))
    # train_txt_x,train_img_x,train_kg_x,train_y = train_txt_x_temp[train_idx],train_img_x_temp[train_idx],train_kg_x_temp[train_idx],train_y_temp[train_idx]
    # val_txt_x, val_img_x, val_kg_x, val_y = train_txt_x_temp[val_idx], train_img_x_temp[val_idx], \
    #                                                 train_kg_x_temp[val_idx], train_y_temp[val_idx]

    # return text_id, tweet, image_id, label, mids_all, \
    #        train_txt_x, train_img_x, train_kg_x, train_y, \
    #        val_txt_x, val_img_x, val_kg_x, val_y, \
    #        test_txt_x, test_img_x, test_kg_x, test_y
    return X_txt, X_img, X_kg, y, y_img

def pre_training_en_full_image(pathset,config):
    text_id_train = []  # 592595287815757825\t\t 这种格式的
    tweet_train = []
    image_id_train = []
    label_train = []  # fake\n 这种格式的
    img_label_train = []
    mid_train = []
    with open(pathset.path_txt_data_train, 'r', encoding='utf-8')as f:
        reader = csv.reader(f)
        for line in reader:
            img = line[2].strip('\t')
            if img != 'none':
                text_id_train.append(line[0].strip('\t').strip('\ufeff').strip('"').strip('\t'))
                tweet_train.append(line[1].strip('\t'))
                image_id_train.append(line[2].strip('\t'))
                label_train.append(int(line[3].strip('\t')))
                # mid = [m.strip(' ').strip('\'') for m in line[4].strip('\t').strip('[').strip(']').split(',')]
                mid_train.append(line[4].strip('\t'))
    # print(len(text_id),len(tweet),len(image_id),len(label)) 总数据16417个
    # print(label)
    text_id_test = []  # 592595287815757825\t\t 这种格式的
    tweet_test = []
    image_id_test = []
    label_test = []  # fake\n 这种格式的
    img_label_test = []
    mid_test = []
    with open(pathset.path_txt_data_test, 'r', encoding='utf-8')as f:
        reader = csv.reader(f)
        for line in reader:
            img = line[2].strip('\t')
            if img != 'none':
                text_id_test.append(line[0].strip('\t').strip('\ufeff').strip('"').strip('\t'))
                tweet_test.append(line[1].strip('\t'))
                image_id_test.append(line[2].strip('\t'))
                label_test.append(int(line[3].strip('\t')))
                # mid = [m.strip(' ').strip('\'') for m in line[4].strip('\t').strip('[').strip(']').split(',')]
                mid_test.append(line[4].strip('\t'))

    UNCASED = pathset.path_bert
    VOCAB = pathset.VOCAB
    tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
    tokens_ids_train = []
    tokens_ids_test = []
    string_process = StringProcess()
    for i in range(len(tweet_train)):
        sentence = string_process.clean_str_BERT(tweet_train[i])
        tokenizer_encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', \
                                       truncation=True, max_length=config.sen_len)
        tokens_ids_train.append(tokenizer_encoding)

    for i in range(len(tweet_test)):
        sentence = string_process.clean_str_BERT(tweet_test[i])
        tokenizer_encoding = tokenizer(sentence, return_tensors='pt', padding='max_length',\
                                       truncation=True, max_length=config.sen_len)
        tokens_ids_test.append(tokenizer_encoding)

    for idx in image_id_train:
        if idx == 'none':
            img_label_train.append(0)
        else:
            img_label_train.append(1)
    for idx in image_id_test:
        if idx == 'none':
            img_label_test.append(0)
        else:
            img_label_test.append(1)

    # 加个断言
    for lbl in img_label_train:
        assert lbl == 1
    # 加个断言
    for lbl in img_label_test:
        assert lbl == 1

    mids_train_all = []
    for i in range(len(text_id_train)):
        mid = []
        # print(i)
        # print(image_id_train[i])
        # if image_id_train[i] in img2mid:
        #     img_mid = img2mid[image_id_train[i]]
        # mid = mids_txt[i] + img_mid
        for en in mid_train[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
            # mid.append(en)
        mid = list(set(mid))
        mids_train_all.append(mid)

    mids_test_all = []
    for i in range(len(text_id_test)):
        mid = []
        # print(i)
        # print(image_id_train[i])
        # if image_id_train[i] in img2mid:
        #     img_mid = img2mid[image_id_train[i]]
        # mid = mids_txt[i] + img_mid
        for en in mid_test[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
            # mid.append(en)
        mid = list(set(mid))
        mids_test_all.append(mid)

    tokens_ids = tokens_ids_train + tokens_ids_test
    # max_len = max([len(single) for single in tokens_ids])
    # 划分数据集 以及读取label_index

    X_txt_train = tokens_ids_train
    X_txt_test = tokens_ids_test
    X_img_train = np.array(image_id_train)
    X_img_test = np.array(image_id_test)
    X_kg_train = np.array(mids_train_all)
    X_kg_test = np.array(mids_test_all)
    # X = np.array(tokens_ids_train)
    y_train = np.array(label_train)
    y_test = np.array(label_test)
    y_img_train = np.array(img_label_train)
    y_img_test = np.array(img_label_test)
    # print(X.size)
    # print(y_train.size)


    return X_txt_train, X_img_train, X_kg_train, y_train, y_img_train, \
           X_txt_test, X_img_test, X_kg_test, y_test, y_img_test

def pre_training_zh_full_image(pathset,config):
    text_id = []  # 592595287815757825\t\t 这种格式的
    tweet = []
    image_id = []
    label = []  # fake\n 这种格式的
    img_label = []
    mids = []
    with open(pathset.path_txt_data, 'r', encoding='utf-8')as f:
        reader = csv.reader(f)
        for line in reader:
            # print(line[0])
            img = line[2].strip('\t').strip('\ufeff').strip('"').strip('\t')
            if img != 'none':
                text_id.append(line[0].strip('\t').strip('\ufeff').strip('"').strip('\t'))
                tweet.append(line[1].strip('\t'))
                image_id.append(line[2].strip('\t').strip('\ufeff').strip('"').strip('\t'))
                label.append(int(line[3].strip('\n')))
                mids.append(line[4].strip('\n'))
    # print(label)
    # print(type(label[0]))
    UNCASED = pathset.path_bert
    VOCAB = pathset.VOCAB
    tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
    tokens_ids = []
    # for tw in tweet:
    #     tw = "[CLS] " + tw + " [SEP]"
    #     tkn = tokenizer.tokenize(tw)
    #     tkn_id = tokenizer.convert_tokens_to_ids(tkn)
    #     tokens_ids.append(tkn_id)
    string_process = StringProcess()
    for i in range(len(tweet)):
        sentence = string_process.clean_str_BERT(tweet[i])
        tokenizer_encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', \
                                       truncation=True, max_length=config.sen_len)
        tokens_ids.append(tokenizer_encoding)
    # mids_all = []
    # for mid in mids:
    #     mids_lst = [m.strip(' ').strip('\'') for m in mid.strip('[').strip(']').split(',')]
    #     # print(mids_lst)
    #     mids_all.append(mids_lst)
    for idx in image_id:
        if idx == 'none':
            img_label.append(0)
        else:
            img_label.append(1)

    # 加个断言
    for lbl in img_label:
        assert lbl == 1


    mids_all = []
    for i in range(len(text_id)):
        mid = []
        for en in mids[i].strip('[').strip(']').split(','):
            if en.strip(' ').strip('\'') != str(None):
                mid.append(en.strip(' ').strip('\'').lstrip('\''))
        mid = list(set(mid))
        mids_all.append(mid)
    # 划分数据集 以及读取label_index

    # X_txt = np.array(tokens_ids)
    X_txt = tokens_ids
    X_img = np.array(image_id)
    X_kg = np.array(mids_all)
    # X = np.array(tokens_ids)
    y = np.array(label)
    y_img = np.array(img_label)

    return X_txt, X_img, X_kg, y, y_img

#读取已经划分好的npy文件
def pre_training_pheme_load_npy(pathset, config,args):
    # 保存npy格式数据，之后的实验直接加载.npy文件 直接存取字典
    # embedding = np.array(embedding)
    # # 保存npy文件
    # with open(clean_image_dir + '/image.npy', 'wb') as f:
    #     # root_index_final = np.array(root_index_final)
    #     np.save(f, embedding)
    # # print('finish')
    # train_dict_list = [train_dict_0, train_dict_1, train_dict_2, train_dict_3, train_dict_4]
    # val_dict_list = [val_dict_0, val_dict_1, val_dict_2, val_dict_3, val_dict_4]
    # test_dict_list = [test_dict_0, test_dict_1, test_dict_2, test_dict_3, test_dict_4]

    save_split_path = '../data/{}/full_image/'.format(args.dataset)
    # if not os.path.exists(save_split_path):
    #     os.mkdir(save_split_path)
    # for i in range(5):
    #     np.save(save_split_path + '{}_split_train_{}'.format(self.args.dataset, i), train_dict_list[i])
    #     np.save(save_split_path + '{}_split_val_{}'.format(self.args.dataset, i), val_dict_list[i])
    #     np.save(save_split_path + '{}_split_test_{}'.format(self.args.dataset, i), test_dict_list[i])
    # npy文件读取
    # dict = {'a':{1,2,3},'b':{4,5,6}}
    train_dict_load = np.load(save_split_path+'full_imagepheme_split_train_2.npy',allow_pickle=True)
    val_dict_load = np.load(save_split_path + 'full_imagepheme_split_val_2.npy', allow_pickle=True)
    test_dict_load = np.load(save_split_path + 'full_imagepheme_split_test_2.npy', allow_pickle=True)
    # dict_load.item()
    # dict_load.item()['a']
    train_dict, val_dict, test_dict = {}, {}, {}

    names_dict_list = ['input_ids','attention_masks','image','vec_1','vec_2','sim_list','y','y_img']
    for name in names_dict_list:
        train_dict[name] = train_dict_load.item()[name]
        val_dict[name] = val_dict_load.item()[name]
        test_dict[name] = test_dict_load.item()[name]

    print('train_dict type:',type(train_dict)) #dict
    print('train_dict inputs_ids type:', type(train_dict['input_ids'])) #list
    # print('train_dict inputs_ids:', train_dict['input_ids'])



    return train_dict,val_dict,test_dict

def pre_training_en_load_npy(pathset,config,args):
    # 保存npy格式数据，之后的实验直接加载.npy文件 直接存取字典
    # embedding = np.array(embedding)
    # # 保存npy文件
    # with open(clean_image_dir + '/image.npy', 'wb') as f:
    #     # root_index_final = np.array(root_index_final)
    #     np.save(f, embedding)
    # # print('finish')
    # train_dict_list = [train_dict_0, train_dict_1, train_dict_2, train_dict_3, train_dict_4]
    # val_dict_list = [val_dict_0, val_dict_1, val_dict_2, val_dict_3, val_dict_4]
    # test_dict_list = [test_dict_0, test_dict_1, test_dict_2, test_dict_3, test_dict_4]

    save_split_path = '../data/{}/full_image/'.format(args.dataset)
    # if not os.path.exists(save_split_path):
    #     os.mkdir(save_split_path)
    # for i in range(5):
    #     np.save(save_split_path + '{}_split_train_{}'.format(self.args.dataset, i), train_dict_list[i])
    #     np.save(save_split_path + '{}_split_val_{}'.format(self.args.dataset, i), val_dict_list[i])
    #     np.save(save_split_path + '{}_split_test_{}'.format(self.args.dataset, i), test_dict_list[i])
    # npy文件读取
    # dict = {'a':{1,2,3},'b':{4,5,6}}
    train_dict_load = np.load(save_split_path + 'full_imagetwitter_split_train_2.npy', allow_pickle=True)
    val_dict_load = np.load(save_split_path + 'full_imagetwitter_split_val_2.npy', allow_pickle=True)
    test_dict_load = np.load(save_split_path + 'full_imagetwitter_split_test_2.npy', allow_pickle=True)
    # dict_load.item()
    # dict_load.item()['a']
    train_dict, val_dict, test_dict = {}, {}, {}

    # names_dict = {'input_ids': input_ids, 'attention_masks': attention_masks, 'image': self.images, \
    #               'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img}
    names_dict_list = ['input_ids', 'attention_masks', 'image', 'vec_1', 'vec_2', 'sim_list', 'y', 'y_img']
    for name in names_dict_list:
        train_dict[name] = train_dict_load.item()[name]
        val_dict[name] = val_dict_load.item()[name]
        test_dict[name] = test_dict_load.item()[name]

    print('train_dict type:', type(train_dict))
    print('train_dict inputs_ids type:', type(train_dict['input_ids']))
    # print('train_dict inputs_ids:', train_dict['input_ids'])

    return train_dict, val_dict, test_dict

def pre_training_zh_load_npy(pathset,config,args):
    # 保存npy格式数据，之后的实验直接加载.npy文件 直接存取字典
    # embedding = np.array(embedding)
    # # 保存npy文件
    # with open(clean_image_dir + '/image.npy', 'wb') as f:
    #     # root_index_final = np.array(root_index_final)
    #     np.save(f, embedding)
    # # print('finish')
    # train_dict_list = [train_dict_0, train_dict_1, train_dict_2, train_dict_3, train_dict_4]
    # val_dict_list = [val_dict_0, val_dict_1, val_dict_2, val_dict_3, val_dict_4]
    # test_dict_list = [test_dict_0, test_dict_1, test_dict_2, test_dict_3, test_dict_4]

    save_split_path = '../data/{}/full_image/'.format(args.dataset)
    # if not os.path.exists(save_split_path):
    #     os.mkdir(save_split_path)
    # for i in range(5):
    #     np.save(save_split_path + '{}_split_train_{}'.format(self.args.dataset, i), train_dict_list[i])
    #     np.save(save_split_path + '{}_split_val_{}'.format(self.args.dataset, i), val_dict_list[i])
    #     np.save(save_split_path + '{}_split_test_{}'.format(self.args.dataset, i), test_dict_list[i])
    # npy文件读取
    # dict = {'a':{1,2,3},'b':{4,5,6}}
    train_dict_load = np.load(save_split_path + 'weibo_split_train_4.npy', allow_pickle=True)
    val_dict_load = np.load(save_split_path + 'weibo_split_val_4.npy', allow_pickle=True)
    test_dict_load = np.load(save_split_path + 'weibo_split_test_4.npy', allow_pickle=True)
    # dict_load.item()
    # dict_load.item()['a']
    train_dict, val_dict, test_dict = {}, {}, {}

    # names_dict = {'input_ids': input_ids, 'attention_masks': attention_masks, 'image': self.images, \
    #               'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img}
    names_dict_list = ['input_ids', 'attention_masks', 'image', 'vec_1', 'vec_2', 'sim_list', 'y', 'y_img']
    for name in names_dict_list:
        train_dict[name] = train_dict_load.item()[name]
        val_dict[name] = val_dict_load.item()[name]
        test_dict[name] = test_dict_load.item()[name]

    print('train_dict type:', type(train_dict))
    print('train_dict inputs_ids type:', type(train_dict['input_ids']))
    # print('train_dict inputs_ids:', train_dict['input_ids'])

    return train_dict, val_dict, test_dict