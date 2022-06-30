import os
import sys
import time

import random
from tqdm import tqdm
import argparse
import pandas as pd
import csv
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable,Function

from util import pre_training_pheme,pre_training_zh,pre_training_en, pre_training_pheme_full_image, pre_training_zh_full_image, pre_training_en_full_image
from configs import inconsistency_Config
from path import path_set_BERT
from models import inconsistency_model_initial,inconsistency_model_weight,inconsistency_model_adversarial, inconsistency_model_replace, inconsistency_model_add_token, inconsistency_model_kdin
from data_process import data_preprocess_ATT_bert_nfold,data_preprocess_ATT_bert, data_preprocess_ATT_bert_nfold_random, data_preprocess_ATT_bert_random,data_preprocess_ATT_bert_nfold_random_full_image, data_preprocess_ATT_bert_random_full_image
from data_load import Dataset_all,Dataset_lower
from loss import Orth_Loss
# from ../process.earlystopping import *
import sys
#sys.path.append("../process/")
from earlystopping import *
import warnings
warnings.filterwarnings('ignore')


# def to_var(x):
#     return Variable(x)

def evaluation(outputs,labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs,labels)).item()
    return correct

def accuracy(pred, targ):
    # pred = torch.max(pred, 1)[1]
    acc = ((pred == targ).float()).sum().item() / targ.size()[0]

    return acc

def macro_f1(pred, targ, num_classes=None):
    # pred = torch.max(pred, 1)[1]
    tp_out = []
    fp_out = []
    fn_out = []
    if num_classes is None:
        num_classes = sorted(set(targ.cpu().numpy().tolist()))
    else:
        num_classes = range(num_classes)
    for i in num_classes:
        tp = ((pred == i) & (targ == i)).sum().item()  # 预测为i，且标签的确为i的
        fp = ((pred == i) & (targ != i)).sum().item()  # 预测为i，但标签不是为i的
        fn = ((pred != i) & (targ == i)).sum().item()  # 预测不是i，但标签是i的
        tp_out.append(tp)
        fp_out.append(fp)
        fn_out.append(fn)

    eval_tp = np.array(tp_out)
    eval_fp = np.array(fp_out)
    eval_fn = np.array(fn_out)

    precision = eval_tp / (eval_tp + eval_fp)
    precision[np.isnan(precision)] = 0
    precision_real = precision[0]
    precision_fake = precision[1]
    precision = np.mean(precision)

    recall = eval_tp / (eval_tp + eval_fn)
    recall[np.isnan(recall)] = 0
    recall_real = recall[0]
    recall_fake = recall[1]
    recall = np.mean(recall)

    f1 = 2 * (precision * recall) / (precision + recall)
    f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real)
    f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake)
    return f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake

def train(train_dict,val_dict,test_dict,model_mode,device,config,dataset,args,transform,pathset,p):
    # data
    if model_mode == 'weight':
        model = inconsistency_model_weight(config=config, pathset=pathset)
    elif model_mode == 'adversarial':
        model = inconsistency_model_adversarial(config=config,pathset=pathset)
    # elif model_mode == 'original':
    #     pass
    model = model.to(device)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)

    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total number of parameters:',pytorch_total_trainable_params)
    # print(model.module)
    #=============================================================================================
    # 'input_ids': input_ids_tensor, 'attention_masks': attention_masks_tensors, 'image': self.images, \
    #         'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img
    train_x_input_ids, train_x_attention_masks, train_x_img,\
    train_x_kg_1, train_x_kg_2, train_x_kg_sim,\
    train_y,train_y_img = train_dict['input_ids'],train_dict['attention_masks'],train_dict['image'],\
                          train_dict['vec_1'],train_dict['vec_2'],train_dict['sim_list'],\
                          train_dict['y'],train_dict['y_img']

    val_x_input_ids, val_x_attention_masks, val_x_img, \
    val_x_kg_1, val_x_kg_2, val_x_kg_sim, \
    val_y, val_y_img = val_dict['input_ids'], val_dict['attention_masks'], val_dict['image'], \
                           val_dict['vec_1'], val_dict['vec_2'], val_dict['sim_list'], \
                           val_dict['y'], val_dict['y_img']

    train_dataset = Dataset_all(x_txt_input_ids=train_x_input_ids, \
                                     x_txt_attention_masks=train_x_attention_masks, \
                                     x_img=train_x_img, x_kg1=train_x_kg_1, x_kg2=train_x_kg_2, \
                                     x_kg_sim=train_x_kg_sim, y=train_y, y_img=train_y_img, transform=transform, pathset=pathset)

    val_dataset = Dataset_all(x_txt_input_ids=val_x_input_ids, \
                                   x_txt_attention_masks=val_x_attention_masks, \
                                   x_img=val_x_img, x_kg1=val_x_kg_1, x_kg2=val_x_kg_2, \
                                   x_kg_sim=val_x_kg_sim, y=val_y, y_img=val_y_img, transform=transform, pathset=pathset)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               # sampler=train_sampler,
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch,
                                             shuffle=True,
                                             # sampler=train_sampler,
                                             num_workers=2)
    t_batch = len(train_loader)
    v_batch = len(val_loader)
    #======================================================================
    model.train()
    criterion_clf = nn.BCELoss()
    criterion_orth = Orth_Loss()
    bert_params = list(map(id,model.module.txtenc.bert.parameters()))
    base_params = filter(lambda p:id(p) not in bert_params,model.parameters())
    # optimizier = optim.Adam(model.parameters(),lr=args.lr)
    optimizier = optim.Adam([
        {'params':model.module.txtenc.bert.parameters(),'lr':5e-5},
        {'params':base_params}
    ],lr = args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizier,'min',factor=0.1,patience=5,verbose=True)
    earlystopping = EarlyStopping(dataset,p,10)
    val_loss_min = 100
    val_acc_max = 0
    # count = 0
    for epoch in range(args.epoch):
        # count_train = 0
        total_loss, total_acc = 0,0
        for i,(input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(train_loader):
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_masks = attention_masks.to(device, dtype=torch.long)
            img = img.to(device)
            kg1 = kg1.to(device, dtype=torch.float)
            kg2 = kg2.to(device,dtype=torch.float)
            kg_sim = kg_sim.to(device, dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            labels_img = labels_img.to(device, dtype=torch.float)
            optimizier.zero_grad()
            outputs_class,output_modal = model(input_ids,attention_masks,img,kg1,kg2,kg_sim,args)
            outputs_class = outputs_class.squeeze()
            output_modal = output_modal.squeeze()
            loss_class = criterion_clf(outputs_class,labels)
            loss_modal = criterion_clf(output_modal,labels_img)
            loss = loss_class + args.lambda_orth*criterion_orth(p_img=model.module.ln_uniq_img.weight,\
                                                   p_txt=model.module.ln_uniq_txt.weight,\
                                                   w_shr=model.module.ln_shr.weight) + args.lambda_m*loss_modal
            loss.backward()
            optimizier.step()
            correct = evaluation(outputs_class,labels)
            total_acc += (correct/args.batch)
            total_loss += loss.item()
            print('the running process is [{}]'.format(p),'[Epoch{}]'.format(epoch+1),\
                  "{}/{}".format(i+1,t_batch),"loss:",loss.item(),'loss_class:',loss_class.item(),'loss_modal:',loss_modal.item(),"acc:",correct*100/args.batch)
        print('\nTrain | Loss:{:.5f} ACC:{:.3f}'.format(total_loss/t_batch, total_acc*100/t_batch))

        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0,0
            for i, (input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(val_loader):
                input_ids = input_ids.to(device, dtype=torch.long)
                attention_masks = attention_masks.to(device, dtype=torch.long)
                img = img.to(device)
                kg1 = kg1.to(device, dtype=torch.float)
                kg2 = kg2.to(device, dtype=torch.float)
                kg_sim = kg_sim.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                labels_img = labels_img.to(device, dtype=torch.float)
                outputs_class, output_modal = model(input_ids, attention_masks, img, kg1, kg2, kg_sim,args)
                outputs_class = outputs_class.squeeze()
                output_modal = output_modal.squeeze()
                loss_class = criterion_clf(outputs_class, labels)
                loss_modal = criterion_clf(output_modal, labels_img)
                loss = loss_class + args.lambda_orth * criterion_orth(p_img=model.module.ln_uniq_img.weight, \
                                                                      p_txt=model.module.ln_uniq_txt.weight, \
                                                                      w_shr=model.module.ln_shr.weight) + args.lambda_m*loss_modal
                correct = evaluation(outputs_class, labels)
                total_acc += (correct/args.batch)
                total_loss += loss.item()
            print("valid | Loss:{:.5f} ACC:{:.3f}".format(total_loss/v_batch, total_acc*100/v_batch))
            val_loss = total_loss/v_batch
            #------------------------earlystopping------------------------------
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                earlystopping.save_checkpoint(total_loss/v_batch, model)
        model.train()
        scheduler.step(val_loss)
    model = earlystopping.load_model()
    test_result_dict = test(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p)
    return test_result_dict

def train_step(train_dict,val_dict,test_dict,model_mode,device,config,dataset,args,transform,pathset,p):
    # data
    if model_mode == 'weight':
        model = inconsistency_model_weight(config=config, pathset=pathset)
    elif model_mode == 'adversarial_step':
        model = inconsistency_model_adversarial(config=config,pathset=pathset)
    # elif model_mode == 'original':
    #     pass
    model = model.to(device)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)

    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total number of parameters:',pytorch_total_trainable_params)
    # print(model.module)
    #=============================================================================================
    # 'input_ids': input_ids_tensor, 'attention_masks': attention_masks_tensors, 'image': self.images, \
    #         'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img
    train_x_input_ids, train_x_attention_masks, train_x_img,\
    train_x_kg_1, train_x_kg_2, train_x_kg_sim,\
    train_y,train_y_img = train_dict['input_ids'],train_dict['attention_masks'],train_dict['image'],\
                          train_dict['vec_1'],train_dict['vec_2'],train_dict['sim_list'],\
                          train_dict['y'],train_dict['y_img']

    val_x_input_ids, val_x_attention_masks, val_x_img, \
    val_x_kg_1, val_x_kg_2, val_x_kg_sim, \
    val_y, val_y_img = val_dict['input_ids'], val_dict['attention_masks'], val_dict['image'], \
                           val_dict['vec_1'], val_dict['vec_2'], val_dict['sim_list'], \
                           val_dict['y'], val_dict['y_img']

    train_dataset = Dataset_all(x_txt_input_ids=train_x_input_ids, \
                                     x_txt_attention_masks=train_x_attention_masks, \
                                     x_img=train_x_img, x_kg1=train_x_kg_1, x_kg2=train_x_kg_2, \
                                     x_kg_sim=train_x_kg_sim, y=train_y, y_img=train_y_img, transform=transform, pathset=pathset)

    val_dataset = Dataset_all(x_txt_input_ids=val_x_input_ids, \
                                   x_txt_attention_masks=val_x_attention_masks, \
                                   x_img=val_x_img, x_kg1=val_x_kg_1, x_kg2=val_x_kg_2, \
                                   x_kg_sim=val_x_kg_sim, y=val_y, y_img=val_y_img, transform=transform, pathset=pathset)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               # sampler=train_sampler,
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch,
                                             shuffle=True,
                                             # sampler=train_sampler,
                                             num_workers=2)
    t_batch = len(train_loader)
    v_batch = len(val_loader)
    #======================================================================
    model.train()
    criterion_clf = nn.BCELoss()
    criterion_orth = Orth_Loss()
    bert_params = list(map(id,model.module.txtenc.bert.parameters()))
    base_params = filter(lambda p:id(p) not in bert_params,model.parameters())
    # optimizier = optim.Adam(model.parameters(),lr=args.lr)
    optimizier = optim.Adam([
        {'params':model.module.txtenc.bert.parameters(),'lr':5e-5},
        {'params':base_params}
    ],lr = args.lr)

    # base_params_2 = []
    # for name, params in model.named_parameters():
    #     if params not in  bert_params:
    #         base_params_2.append(params)
    # print('base_params:', type(base_params), base_params)#<class 'filter'> <filter object at 0x7fdd8b73c0f0>
    # optimizier2 = optim.Adam(base_params_2, lr= args.lr)
    optimizier2 = optim.Adam([{'params': base_params}], lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizier,'min',factor=0.1,patience=5,verbose=True)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizier2,'min',factor=0.1,patience=5,verbose=True)
    earlystopping = EarlyStopping(dataset,p,10)
    val_loss_min = 100
    val_acc_max = 0
    # count = 0
    for epoch in range(args.epoch):
        # count_train = 0
        total_loss, total_acc = 0,0
        for i,(input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(train_loader):
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_masks = attention_masks.to(device, dtype=torch.long)
            img = img.to(device)
            kg1 = kg1.to(device, dtype=torch.float)
            kg2 = kg2.to(device,dtype=torch.float)
            kg_sim = kg_sim.to(device, dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            labels_img = labels_img.to(device, dtype=torch.float)

            outputs_class,output_modal = model(input_ids,attention_masks,img,kg1,kg2,kg_sim,args)
            outputs_class = outputs_class.squeeze()
            output_modal = output_modal.squeeze()
            loss_class = criterion_clf(outputs_class,labels)
            loss_modal = criterion_clf(output_modal,labels_img)
            loss = loss_class + args.lambda_orth*criterion_orth(p_img=model.module.ln_uniq_img.weight,\
                                                   p_txt=model.module.ln_uniq_txt.weight,\
                                                   w_shr=model.module.ln_shr.weight) + args.lambda_m*loss_modal
            if epoch < 5:
                optimizier.zero_grad()
                loss.backward()
                optimizier.step()
            else:
                optimizier2.zero_grad()
                loss.backward()
                optimizier2.step()
            correct = evaluation(outputs_class,labels)
            total_acc += (correct/args.batch)
            total_loss += loss.item()
            print('the running process is [{}]'.format(p),'[Epoch{}]'.format(epoch+1),\
                  "{}/{}".format(i+1,t_batch),"loss:",loss.item(),'loss_class:',loss_class.item(),'loss_modal:',loss_modal.item(),"acc:",correct*100/args.batch)
        print('\nTrain | Loss:{:.5f} ACC:{:.3f}'.format(total_loss/t_batch, total_acc*100/t_batch))

        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0,0
            for i, (input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(val_loader):
                input_ids = input_ids.to(device, dtype=torch.long)
                attention_masks = attention_masks.to(device, dtype=torch.long)
                img = img.to(device)
                kg1 = kg1.to(device, dtype=torch.float)
                kg2 = kg2.to(device, dtype=torch.float)
                kg_sim = kg_sim.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                labels_img = labels_img.to(device, dtype=torch.float)
                outputs_class, output_modal = model(input_ids, attention_masks, img, kg1, kg2, kg_sim,args)
                outputs_class = outputs_class.squeeze()
                output_modal = output_modal.squeeze()
                loss_class = criterion_clf(outputs_class, labels)
                loss_modal = criterion_clf(output_modal, labels_img)
                loss = loss_class + args.lambda_orth * criterion_orth(p_img=model.module.ln_uniq_img.weight, \
                                                                      p_txt=model.module.ln_uniq_txt.weight, \
                                                                      w_shr=model.module.ln_shr.weight) + args.lambda_m*loss_modal
                correct = evaluation(outputs_class, labels)
                total_acc += (correct/args.batch)
                total_loss += loss.item()
            print("valid | Loss:{:.5f} ACC:{:.3f}".format(total_loss/v_batch, total_acc*100/v_batch))
            val_loss = total_loss/v_batch
            #------------------------earlystopping------------------------------
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                earlystopping.save_checkpoint(total_loss/v_batch, model)
        model.train()
        if epoch < 5:
            scheduler.step(val_loss)
        else:
            scheduler2.step(val_loss)
    model = earlystopping.load_model()
    test_result_dict = test(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p)
    return test_result_dict

def test(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p):
    test_result_dict = {}
    test_x_input_ids, test_x_attention_masks, test_x_img, \
    test_x_kg_1, test_x_kg_2, test_x_kg_sim, \
    test_y, test_y_img = test_dict['input_ids'], test_dict['attention_masks'], test_dict['image'], \
                       test_dict['vec_1'], test_dict['vec_2'], test_dict['sim_list'], \
                       test_dict['y'], test_dict['y_img']

    test_dataset = Dataset_all(x_txt_input_ids=test_x_input_ids, \
                                          x_txt_attention_masks=test_x_attention_masks, \
                                          x_img=test_x_img, x_kg1=test_x_kg_1, x_kg2=test_x_kg_2, \
                                          x_kg_sim=test_x_kg_sim, y=test_y, y_img=test_y_img, transform=transform,
                                          pathset=pathset)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=args.batch,
                                             shuffle=False,
                                             # sampler=train_sampler,
                                             num_workers=2)
    model.eval()
    with torch.no_grad():
        for i, (input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(test_loader):
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_masks = attention_masks.to(device, dtype=torch.long)
            img = img.to(device)
            kg1 = kg1.to(device, dtype=torch.float)
            kg2 = kg2.to(device, dtype=torch.float)
            kg_sim = kg_sim.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            labels_img = labels_img.to(device, dtype=torch.float)
            outputs_class, output_modal = model(input_ids, attention_masks, img, kg1, kg2, kg_sim,args)
            outputs_class = outputs_class.squeeze()
            output_modal = output_modal.squeeze()
            outputs_class[outputs_class >= 0.5] = 1
            outputs_class[outputs_class < 0.5] = 0
            if i == 0:
                outputs_class_all = outputs_class
                labels_all = labels
            else:
                outputs_class_all = torch.cat([outputs_class_all,outputs_class],dim=0)
                labels_all = torch.cat([labels_all,labels],dim=0)
    acc = accuracy(outputs_class_all,labels_all)
    f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake = macro_f1(
        outputs_class_all, labels_all, num_classes=2)
    print('----------------------------------------')
    print('acc:',acc,'prec:',precision,'rec:',recall,'f1:',f1)
    print('prec-fake:', precision_fake, 'rec-fake:', recall_fake, 'f1-fake:', f1_fake)
    print('prec-real:', precision_real, 'rec-real:', recall_real, 'f1-real:', f1_real)


    test_result_dict['acc'] = acc
    test_result_dict['prec'] = precision
    test_result_dict['rec'] = recall
    test_result_dict['f1'] = f1

    # test_dict['acc'] = acc
    test_result_dict['prec_fake'] = precision_fake
    test_result_dict['rec_fake'] = recall_fake
    test_result_dict['f1_fake'] = f1_fake

    test_result_dict['prec_real'] = precision_real
    test_result_dict['rec_real'] = recall_real
    test_result_dict['f1_real'] = f1_real

    with open('../result/result_test_{}_{}_batch{}_lr{}_orth{}_modal{}_{}.txt'.format(dataset,model_mode,args.batch, args.lr, args.lambda_orth, args.lambda_m,p),'w',encoding='utf-8',newline='')as f:
        string1 = 'acc:'+str(acc)+'\t'+'f1:'+str(f1)+'\t'+'precision:'+str(precision)+'\t'+'recall:'+str(recall)+'\n'
        string2 = 'f1_real:' + str(f1_real) + '\t' + 'precision_real:' + str(
            precision_real) + '\t' + 'recall_real:' + str(recall_real)+'\n'
        string3 = 'f1_fake:' + str(f1_fake) + '\t' + 'precision_fake:' + str(
            precision_fake) + '\t' + 'recall_fake:' + str(recall_fake)+'\n\n\n'
        f.writelines(string1)
        f.writelines(string2)
        f.writelines(string3)

    return test_result_dict


def train_initial(train_dict,val_dict,test_dict,model_mode,device,config,dataset,args,transform,pathset,p):
    # data
    if model_mode == 'initial':
        model = inconsistency_model_initial(config=config,pathset=pathset)

    # elif model_mode == 'original':
    #     pass
    model = model.to(device)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)

    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total number of parameters:',pytorch_total_trainable_params)
    # print(model.module)
    #=============================================================================================
    # 'input_ids': input_ids_tensor, 'attention_masks': attention_masks_tensors, 'image': self.images, \
    #         'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img
    train_x_input_ids, train_x_attention_masks, train_x_img,\
    train_x_kg_1, train_x_kg_2, train_x_kg_sim,\
    train_y,train_y_img = train_dict['input_ids'],train_dict['attention_masks'],train_dict['image'],\
                          train_dict['vec_1'],train_dict['vec_2'],train_dict['sim_list'],\
                          train_dict['y'],train_dict['y_img']

    val_x_input_ids, val_x_attention_masks, val_x_img, \
    val_x_kg_1, val_x_kg_2, val_x_kg_sim, \
    val_y, val_y_img = val_dict['input_ids'], val_dict['attention_masks'], val_dict['image'], \
                           val_dict['vec_1'], val_dict['vec_2'], val_dict['sim_list'], \
                           val_dict['y'], val_dict['y_img']

    train_dataset = Dataset_all(x_txt_input_ids=train_x_input_ids, \
                                     x_txt_attention_masks=train_x_attention_masks, \
                                     x_img=train_x_img, x_kg1=train_x_kg_1, x_kg2=train_x_kg_2, \
                                     x_kg_sim=train_x_kg_sim, y=train_y, y_img=train_y_img, transform=transform, pathset=pathset)

    val_dataset = Dataset_all(x_txt_input_ids=val_x_input_ids, \
                                   x_txt_attention_masks=val_x_attention_masks, \
                                   x_img=val_x_img, x_kg1=val_x_kg_1, x_kg2=val_x_kg_2, \
                                   x_kg_sim=val_x_kg_sim, y=val_y, y_img=val_y_img, transform=transform, pathset=pathset)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               # sampler=train_sampler,
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch,
                                             shuffle=True,
                                             # sampler=train_sampler,
                                             num_workers=2)
    t_batch = len(train_loader)
    v_batch = len(val_loader)
    #======================================================================
    model.train()
    criterion_clf = nn.BCELoss()
    criterion_orth = Orth_Loss()
    bert_params = list(map(id,model.module.txtenc.bert.parameters()))
    base_params = filter(lambda p:id(p) not in bert_params,model.parameters())
    # optimizier = optim.Adam(model.parameters(),lr=args.lr)
    optimizier = optim.Adam([
        {'params':model.module.txtenc.bert.parameters(),'lr':5e-5},
        {'params':base_params}
    ],lr = args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizier,'min',factor=0.1,patience=5,verbose=True)
    earlystopping = EarlyStopping(dataset,p,10)
    val_loss_min = 100
    val_acc_max = 0
    # count = 0
    for epoch in range(args.epoch):
        # count_train = 0
        total_loss, total_acc = 0,0
        for i,(input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(train_loader):
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_masks = attention_masks.to(device, dtype=torch.long)
            img = img.to(device)
            kg1 = kg1.to(device, dtype=torch.float)
            kg2 = kg2.to(device,dtype=torch.float)
            kg_sim = kg_sim.to(device, dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            labels_img = labels_img.to(device, dtype=torch.float)
            optimizier.zero_grad()
            outputs_class,output_modal = model(input_ids,attention_masks,img,kg1,kg2,kg_sim, labels_img,args)
            outputs_class = outputs_class.squeeze()
            output_modal = output_modal.squeeze()
            loss_class = criterion_clf(outputs_class,labels)
            loss_modal = criterion_clf(output_modal,labels_img)
            loss = loss_class + args.lambda_orth*criterion_orth(p_img=model.module.ln_uniq_img.weight,\
                                                   p_txt=model.module.ln_uniq_txt.weight,\
                                                   w_shr=model.module.ln_shr.weight) + args.lambda_m*loss_modal
            loss.backward()
            optimizier.step()
            correct = evaluation(outputs_class,labels)
            total_acc += (correct/args.batch)
            total_loss += loss.item()
            print('the running process is [{}]'.format(p),'[Epoch{}]'.format(epoch+1),\
                  "{}/{}".format(i+1,t_batch),"loss:",loss.item(),'loss_class:',loss_class.item(),'loss_modal:',loss_modal.item(),"acc:",correct*100/args.batch)
        print('\nTrain | Loss:{:.5f} ACC:{:.3f}'.format(total_loss/t_batch, total_acc*100/t_batch))

        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0,0
            for i, (input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(val_loader):
                input_ids = input_ids.to(device, dtype=torch.long)
                attention_masks = attention_masks.to(device, dtype=torch.long)
                img = img.to(device)
                kg1 = kg1.to(device, dtype=torch.float)
                kg2 = kg2.to(device, dtype=torch.float)
                kg_sim = kg_sim.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                labels_img = labels_img.to(device, dtype=torch.float)
                outputs_class, output_modal = model(input_ids, attention_masks, img, kg1, kg2, kg_sim, labels_img,args)
                outputs_class = outputs_class.squeeze()
                output_modal = output_modal.squeeze()
                loss_class = criterion_clf(outputs_class, labels)
                loss_modal = criterion_clf(output_modal, labels_img)
                loss = loss_class + args.lambda_orth * criterion_orth(p_img=model.module.ln_uniq_img.weight, \
                                                                      p_txt=model.module.ln_uniq_txt.weight, \
                                                                      w_shr=model.module.ln_shr.weight) + args.lambda_m*loss_modal
                correct = evaluation(outputs_class, labels)
                total_acc += (correct/args.batch)
                total_loss += loss.item()
            print("valid | Loss:{:.5f} ACC:{:.3f}".format(total_loss/v_batch, total_acc*100/v_batch))
            val_loss = total_loss/v_batch
            #------------------------earlystopping------------------------------
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                earlystopping.save_checkpoint(total_loss/v_batch, model)
        model.train()
        scheduler.step(val_loss)
    model = earlystopping.load_model()
    test_result_dict = test_initial(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p)
    return test_result_dict

def train_initial_step(train_dict,val_dict,test_dict,model_mode,device,config,dataset,args,transform,pathset,p):
    # data
    if model_mode == 'initial':
        model = inconsistency_model_initial(config=config,pathset=pathset)

    # elif model_mode == 'original':
    #     pass
    model = model.to(device)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)

    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total number of parameters:',pytorch_total_trainable_params)
    # print(model.module)
    #=============================================================================================
    # 'input_ids': input_ids_tensor, 'attention_masks': attention_masks_tensors, 'image': self.images, \
    #         'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img
    train_x_input_ids, train_x_attention_masks, train_x_img,\
    train_x_kg_1, train_x_kg_2, train_x_kg_sim,\
    train_y,train_y_img = train_dict['input_ids'],train_dict['attention_masks'],train_dict['image'],\
                          train_dict['vec_1'],train_dict['vec_2'],train_dict['sim_list'],\
                          train_dict['y'],train_dict['y_img']

    val_x_input_ids, val_x_attention_masks, val_x_img, \
    val_x_kg_1, val_x_kg_2, val_x_kg_sim, \
    val_y, val_y_img = val_dict['input_ids'], val_dict['attention_masks'], val_dict['image'], \
                           val_dict['vec_1'], val_dict['vec_2'], val_dict['sim_list'], \
                           val_dict['y'], val_dict['y_img']

    train_dataset = Dataset_all(x_txt_input_ids=train_x_input_ids, \
                                     x_txt_attention_masks=train_x_attention_masks, \
                                     x_img=train_x_img, x_kg1=train_x_kg_1, x_kg2=train_x_kg_2, \
                                     x_kg_sim=train_x_kg_sim, y=train_y, y_img=train_y_img, transform=transform, pathset=pathset)

    val_dataset = Dataset_all(x_txt_input_ids=val_x_input_ids, \
                                   x_txt_attention_masks=val_x_attention_masks, \
                                   x_img=val_x_img, x_kg1=val_x_kg_1, x_kg2=val_x_kg_2, \
                                   x_kg_sim=val_x_kg_sim, y=val_y, y_img=val_y_img, transform=transform, pathset=pathset)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               # sampler=train_sampler,
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch,
                                             shuffle=True,
                                             # sampler=train_sampler,
                                             num_workers=2)
    t_batch = len(train_loader)
    v_batch = len(val_loader)
    #======================================================================
    model.train()
    criterion_clf = nn.BCELoss()
    criterion_orth = Orth_Loss()
    bert_params = list(map(id,model.module.txtenc.bert.parameters()))
    base_params = filter(lambda p:id(p) not in bert_params,model.parameters())
    # optimizier = optim.Adam(model.parameters(),lr=args.lr)
    optimizier = optim.Adam([
        {'params':model.module.txtenc.bert.parameters(),'lr':5e-5},
        {'params':base_params}
    ],lr = args.lr)

    # base_params_2 = []
    # for name, params in model.named_parameters():
    #     if params not in  bert_params:
    #         base_params_2.append(params)
    # print('base_params:', type(base_params), base_params)#<class 'filter'> <filter object at 0x7fdd8b73c0f0>
    # optimizier2 = optim.Adam(base_params_2, lr= args.lr)
    optimizier2 = optim.Adam([{'params': base_params}], lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizier,'min',factor=0.1,patience=5,verbose=True)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizier2,'min',factor=0.1,patience=5,verbose=True)
    earlystopping = EarlyStopping(dataset,p,10)
    val_loss_min = 100
    val_acc_max = 0
    # count = 0
    for epoch in range(args.epoch):
        # count_train = 0
        total_loss, total_acc = 0,0
        for i,(input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(train_loader):
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_masks = attention_masks.to(device, dtype=torch.long)
            img = img.to(device)
            kg1 = kg1.to(device, dtype=torch.float)
            kg2 = kg2.to(device,dtype=torch.float)
            kg_sim = kg_sim.to(device, dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            labels_img = labels_img.to(device, dtype=torch.float)

            outputs_class,output_modal = model(input_ids,attention_masks,img,kg1,kg2,kg_sim, labels_img,args)
            outputs_class = outputs_class.squeeze()
            output_modal = output_modal.squeeze()
            loss_class = criterion_clf(outputs_class,labels)
            loss_modal = criterion_clf(output_modal,labels_img)
            loss = loss_class + args.lambda_orth*criterion_orth(p_img=model.module.ln_uniq_img.weight,\
                                                   p_txt=model.module.ln_uniq_txt.weight,\
                                                   w_shr=model.module.ln_shr.weight) + args.lambda_m*loss_modal
            if epoch < 5:
                optimizier.zero_grad()
                loss.backward()
                optimizier.step()
            else:
                optimizier2.zero_grad()
                loss.backward()
                optimizier2.step()
            correct = evaluation(outputs_class,labels)
            total_acc += (correct/args.batch)
            total_loss += loss.item()
            print('the running process is [{}]'.format(p),'[Epoch{}]'.format(epoch+1),\
                  "{}/{}".format(i+1,t_batch),"loss:",loss.item(),'loss_class:',loss_class.item(),'loss_modal:',loss_modal.item(),"acc:",correct*100/args.batch)
        print('\nTrain | Loss:{:.5f} ACC:{:.3f}'.format(total_loss/t_batch, total_acc*100/t_batch))

        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0,0
            for i, (input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(val_loader):
                input_ids = input_ids.to(device, dtype=torch.long)
                attention_masks = attention_masks.to(device, dtype=torch.long)
                img = img.to(device)
                kg1 = kg1.to(device, dtype=torch.float)
                kg2 = kg2.to(device, dtype=torch.float)
                kg_sim = kg_sim.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                labels_img = labels_img.to(device, dtype=torch.float)
                outputs_class, output_modal = model(input_ids, attention_masks, img, kg1, kg2, kg_sim, labels_img,args)
                outputs_class = outputs_class.squeeze()
                output_modal = output_modal.squeeze()
                loss_class = criterion_clf(outputs_class, labels)
                loss_modal = criterion_clf(output_modal, labels_img)
                loss = loss_class + args.lambda_orth * criterion_orth(p_img=model.module.ln_uniq_img.weight, \
                                                                      p_txt=model.module.ln_uniq_txt.weight, \
                                                                      w_shr=model.module.ln_shr.weight) + args.lambda_m*loss_modal
                correct = evaluation(outputs_class, labels)
                total_acc += (correct/args.batch)
                total_loss += loss.item()
            print("valid | Loss:{:.5f} ACC:{:.3f}".format(total_loss/v_batch, total_acc*100/v_batch))
            val_loss = total_loss/v_batch
            #------------------------earlystopping------------------------------
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                earlystopping.save_checkpoint(total_loss/v_batch, model)
        model.train()
        if epoch < 5:
            scheduler.step(val_loss)
        else:
            scheduler2.step(val_loss)
    model = earlystopping.load_model()
    test_result_dict = test_initial(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p)
    return test_result_dict

def test_initial(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p):
    test_result_dict = {}
    test_x_input_ids, test_x_attention_masks, test_x_img, \
    test_x_kg_1, test_x_kg_2, test_x_kg_sim, \
    test_y, test_y_img = test_dict['input_ids'], test_dict['attention_masks'], test_dict['image'], \
                       test_dict['vec_1'], test_dict['vec_2'], test_dict['sim_list'], \
                       test_dict['y'], test_dict['y_img']

    test_dataset = Dataset_all(x_txt_input_ids=test_x_input_ids, \
                                          x_txt_attention_masks=test_x_attention_masks, \
                                          x_img=test_x_img, x_kg1=test_x_kg_1, x_kg2=test_x_kg_2, \
                                          x_kg_sim=test_x_kg_sim, y=test_y, y_img=test_y_img, transform=transform,
                                          pathset=pathset)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=args.batch,
                                             shuffle=False,
                                             # sampler=train_sampler,
                                             num_workers=2)
    model.eval()
    with torch.no_grad():
        for i, (input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(test_loader):
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_masks = attention_masks.to(device, dtype=torch.long)
            img = img.to(device)
            kg1 = kg1.to(device, dtype=torch.float)
            kg2 = kg2.to(device, dtype=torch.float)
            kg_sim = kg_sim.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            labels_img = labels_img.to(device, dtype=torch.float)
            outputs_class, output_modal = model(input_ids, attention_masks, img, kg1, kg2, kg_sim, labels_img,args)
            outputs_class = outputs_class.squeeze()
            output_modal = output_modal.squeeze()
            outputs_class[outputs_class >= 0.5] = 1
            outputs_class[outputs_class < 0.5] = 0
            if i == 0:
                outputs_class_all = outputs_class
                labels_all = labels
            else:
                outputs_class_all = torch.cat([outputs_class_all,outputs_class],dim=0)
                labels_all = torch.cat([labels_all,labels],dim=0)
    acc = accuracy(outputs_class_all,labels_all)
    f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake = macro_f1(
        outputs_class_all, labels_all, num_classes=2)
    print('----------------------------------------')
    print('acc:',acc,'prec:',precision,'rec:',recall,'f1:',f1)
    print('prec-fake:', precision_fake, 'rec-fake:', recall_fake, 'f1-fake:', f1_fake)
    print('prec-real:', precision_real, 'rec-real:', recall_real, 'f1-real:', f1_real)


    test_result_dict['acc'] = acc
    test_result_dict['prec'] = precision
    test_result_dict['rec'] = recall
    test_result_dict['f1'] = f1

    # test_dict['acc'] = acc
    test_result_dict['prec_fake'] = precision_fake
    test_result_dict['rec_fake'] = recall_fake
    test_result_dict['f1_fake'] = f1_fake

    test_result_dict['prec_real'] = precision_real
    test_result_dict['rec_real'] = recall_real
    test_result_dict['f1_real'] = f1_real

    with open('../result/result_test_{}_{}_batch{}_lr{}_orth{}_modal{}_{}.txt'.format(dataset,model_mode,args.batch, args.lr, args.lambda_orth, args.lambda_m,p),'w',encoding='utf-8',newline='')as f:
        string1 = 'acc:'+str(acc)+'\t'+'f1:'+str(f1)+'\t'+'precision:'+str(precision)+'\t'+'recall:'+str(recall)+'\n'
        string2 = 'f1_real:' + str(f1_real) + '\t' + 'precision_real:' + str(
            precision_real) + '\t' + 'recall_real:' + str(recall_real)+'\n'
        string3 = 'f1_fake:' + str(f1_fake) + '\t' + 'precision_fake:' + str(
            precision_fake) + '\t' + 'recall_fake:' + str(recall_fake)+'\n\n\n'
        f.writelines(string1)
        f.writelines(string2)
        f.writelines(string3)

    return test_result_dict


def train_replace(train_dict,val_dict,test_dict,model_mode,device,config,dataset,args,transform,pathset,p):
    # data
    if model_mode == 'kdin':
        model = inconsistency_model_kdin(config=config,pathset=pathset)
    # elif model_mode == 'original':
    #     pass
    if model_mode == 'add_token':
        model = inconsistency_model_add_token(config=config,pathset=pathset)

    model = model.to(device)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)

    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total number of parameters:',pytorch_total_trainable_params)
    # print(model.module)
    #=============================================================================================
    # 'input_ids': input_ids_tensor, 'attention_masks': attention_masks_tensors, 'image': self.images, \
    #         'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img
    train_x_input_ids, train_x_attention_masks, train_x_img,\
    train_x_kg_1, train_x_kg_2, train_x_kg_sim,\
    train_y,train_y_img = train_dict['input_ids'],train_dict['attention_masks'],train_dict['image'],\
                          train_dict['vec_1'],train_dict['vec_2'],train_dict['sim_list'],\
                          train_dict['y'],train_dict['y_img']

    val_x_input_ids, val_x_attention_masks, val_x_img, \
    val_x_kg_1, val_x_kg_2, val_x_kg_sim, \
    val_y, val_y_img = val_dict['input_ids'], val_dict['attention_masks'], val_dict['image'], \
                           val_dict['vec_1'], val_dict['vec_2'], val_dict['sim_list'], \
                           val_dict['y'], val_dict['y_img']

    #判断lower取值，false则采用原数据集，True则采用只有text的数据
    if args.lower == False:
        train_dataset = Dataset_all(x_txt_input_ids=train_x_input_ids, \
                                         x_txt_attention_masks=train_x_attention_masks, \
                                         x_img=train_x_img, x_kg1=train_x_kg_1, x_kg2=train_x_kg_2, \
                                         x_kg_sim=train_x_kg_sim, y=train_y, y_img=train_y_img, transform=transform, pathset=pathset)

        val_dataset = Dataset_all(x_txt_input_ids=val_x_input_ids, \
                                       x_txt_attention_masks=val_x_attention_masks, \
                                       x_img=val_x_img, x_kg1=val_x_kg_1, x_kg2=val_x_kg_2, \
                                       x_kg_sim=val_x_kg_sim, y=val_y, y_img=val_y_img, transform=transform, pathset=pathset)
    elif args.lower == True:
        train_dataset = Dataset_lower(x_txt_input_ids=train_x_input_ids, \
                                    x_txt_attention_masks=train_x_attention_masks, \
                                    x_img=train_x_img, x_kg1=train_x_kg_1, x_kg2=train_x_kg_2, \
                                    x_kg_sim=train_x_kg_sim, y=train_y, y_img=train_y_img, transform=transform,
                                    pathset=pathset)

        val_dataset = Dataset_lower(x_txt_input_ids=val_x_input_ids, \
                                  x_txt_attention_masks=val_x_attention_masks, \
                                  x_img=val_x_img, x_kg1=val_x_kg_1, x_kg2=val_x_kg_2, \
                                  x_kg_sim=val_x_kg_sim, y=val_y, y_img=val_y_img, transform=transform, pathset=pathset)



    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               # sampler=train_sampler,
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch,
                                             shuffle=True,
                                             # sampler=train_sampler,
                                             num_workers=2)
    t_batch = len(train_loader)
    v_batch = len(val_loader)
    #======================================================================
    model.train()
    criterion_clf = nn.BCELoss()
    criterion_orth = Orth_Loss()
    bert_params = list(map(id,model.module.txtenc.bert.parameters()))
    base_params = filter(lambda p:id(p) not in bert_params,model.parameters())
    # optimizier = optim.Adam(model.parameters(),lr=args.lr)
    optimizier = optim.Adam([
        {'params':model.module.txtenc.bert.parameters(),'lr':5e-5},
        {'params':base_params}
    ],lr = args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizier,'min',factor=0.1,patience=5,verbose=True)
    earlystopping = EarlyStopping(dataset,p,10)
    val_loss_min = 100
    val_acc_max = 0
    # count = 0
    for epoch in range(args.epoch):
        # count_train = 0
        total_loss, total_acc = 0,0
        for i,(input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(train_loader):
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_masks = attention_masks.to(device, dtype=torch.long)
            img = img.to(device)
            kg1 = kg1.to(device, dtype=torch.float)
            kg2 = kg2.to(device,dtype=torch.float)
            kg_sim = kg_sim.to(device, dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            labels_img = labels_img.to(device, dtype=torch.float)
            optimizier.zero_grad()
            outputs_class = model(input_ids,attention_masks,img,kg1,kg2,kg_sim,labels_img,args)
            outputs_class = outputs_class.squeeze()

            loss_class = criterion_clf(outputs_class,labels)

            loss = loss_class + args.lambda_orth*criterion_orth(p_img=model.module.ln_uniq_img.weight,\
                                                   p_txt=model.module.ln_uniq_txt.weight,\
                                                   w_shr=model.module.ln_shr.weight)
            loss.backward()
            optimizier.step()
            correct = evaluation(outputs_class,labels)
            total_acc += (correct/args.batch)
            total_loss += loss.item()
            print('the running process is [{}]'.format(p),'[Epoch{}]'.format(epoch+1),\
                  "{}/{}".format(i+1,t_batch),"loss:",loss.item(),'loss_class:',loss_class.item(),"acc:",correct*100/args.batch)
        print('\nTrain | Loss:{:.5f} ACC:{:.3f}'.format(total_loss/t_batch, total_acc*100/t_batch))

        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0,0
            for i, (input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(val_loader):
                input_ids = input_ids.to(device, dtype=torch.long)
                attention_masks = attention_masks.to(device, dtype=torch.long)
                img = img.to(device)
                kg1 = kg1.to(device, dtype=torch.float)
                kg2 = kg2.to(device, dtype=torch.float)
                kg_sim = kg_sim.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                labels_img = labels_img.to(device, dtype=torch.float)
                outputs_class = model(input_ids, attention_masks, img, kg1, kg2, kg_sim,labels_img,args)
                outputs_class = outputs_class.squeeze()

                loss_class = criterion_clf(outputs_class, labels)

                loss = loss_class + args.lambda_orth * criterion_orth(p_img=model.module.ln_uniq_img.weight, \
                                                                      p_txt=model.module.ln_uniq_txt.weight, \
                                                                      w_shr=model.module.ln_shr.weight)
                correct = evaluation(outputs_class, labels)
                total_acc += (correct/args.batch)
                total_loss += loss.item()
            print("valid | Loss:{:.5f} ACC:{:.3f}".format(total_loss/v_batch, total_acc*100/v_batch))
            val_loss = total_loss/v_batch
            #------------------------earlystopping------------------------------
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                earlystopping.save_checkpoint(total_loss/v_batch, model)
        model.train()
        scheduler.step(val_loss)
    model = earlystopping.load_model()
    test_result_dict = test_replace(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p)
    return test_result_dict

def train_replace_step(train_dict,val_dict,test_dict,model_mode,device,config,dataset,args,transform,pathset,p):
    # data
    if model_mode == 'replace_step':
        model = inconsistency_model_replace(config=config,pathset=pathset)
    # elif model_mode == 'original':
    #     pass
    if model_mode == 'add_token_step':
        model = inconsistency_model_add_token(config=config,pathset=pathset)

    model = model.to(device)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)

    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total number of parameters:',pytorch_total_trainable_params)
    # print(model.module)
    #=============================================================================================
    # 'input_ids': input_ids_tensor, 'attention_masks': attention_masks_tensors, 'image': self.images, \
    #         'vec_1': vec_all_1, 'vec_2': vec_all_2, 'sim_list': sim_all, 'y': y, 'y_img': y_img
    train_x_input_ids, train_x_attention_masks, train_x_img,\
    train_x_kg_1, train_x_kg_2, train_x_kg_sim,\
    train_y,train_y_img = train_dict['input_ids'],train_dict['attention_masks'],train_dict['image'],\
                          train_dict['vec_1'],train_dict['vec_2'],train_dict['sim_list'],\
                          train_dict['y'],train_dict['y_img']

    val_x_input_ids, val_x_attention_masks, val_x_img, \
    val_x_kg_1, val_x_kg_2, val_x_kg_sim, \
    val_y, val_y_img = val_dict['input_ids'], val_dict['attention_masks'], val_dict['image'], \
                           val_dict['vec_1'], val_dict['vec_2'], val_dict['sim_list'], \
                           val_dict['y'], val_dict['y_img']

    train_dataset = Dataset_all(x_txt_input_ids=train_x_input_ids, \
                                     x_txt_attention_masks=train_x_attention_masks, \
                                     x_img=train_x_img, x_kg1=train_x_kg_1, x_kg2=train_x_kg_2, \
                                     x_kg_sim=train_x_kg_sim, y=train_y, y_img=train_y_img, transform=transform, pathset=pathset)

    val_dataset = Dataset_all(x_txt_input_ids=val_x_input_ids, \
                                   x_txt_attention_masks=val_x_attention_masks, \
                                   x_img=val_x_img, x_kg1=val_x_kg_1, x_kg2=val_x_kg_2, \
                                   x_kg_sim=val_x_kg_sim, y=val_y, y_img=val_y_img, transform=transform, pathset=pathset)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               # sampler=train_sampler,
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch,
                                             shuffle=True,
                                             # sampler=train_sampler,
                                             num_workers=2)
    t_batch = len(train_loader)
    v_batch = len(val_loader)
    #======================================================================
    model.train()
    criterion_clf = nn.BCELoss()
    criterion_orth = Orth_Loss()
    bert_params = list(map(id,model.module.txtenc.bert.parameters()))
    base_params = filter(lambda p:id(p) not in bert_params,model.parameters())
    # optimizier = optim.Adam(model.parameters(),lr=args.lr)
    optimizier = optim.Adam([
        {'params':model.module.txtenc.bert.parameters(),'lr':5e-5},
        {'params':base_params}
    ],lr = args.lr)

    # print('bert params:',bert_params,type(bert_params))
    # base_params_2 = list(base_params)
    # for params in model.parameters():
    #     print('params in model.parameters()',params)
    #     if params not in bert_params:
    #         base_params_2.append(params)
    # print('base_params:', type(base_params), base_params)#<class 'filter'> <filter object at 0x7fdd8b73c0f0>
    optimizier2 = optim.Adam([{'params':base_params}], lr=args.lr)
    # print('base_params:',type(base_params),base_params)
    # optimizier2 = optim.Adam(base_params,lr = args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizier,'min',factor=0.1,patience=5,verbose=True)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizier2,'min',factor=0.1,patience=5,verbose=True)
    earlystopping = EarlyStopping(dataset,p,10)
    val_loss_min = 100
    val_acc_max = 0
    # count = 0
    for epoch in range(args.epoch):
        # count_train = 0
        total_loss, total_acc = 0,0
        for i,(input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(train_loader):
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_masks = attention_masks.to(device, dtype=torch.long)
            img = img.to(device)
            kg1 = kg1.to(device, dtype=torch.float)
            kg2 = kg2.to(device,dtype=torch.float)
            kg_sim = kg_sim.to(device, dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            labels_img = labels_img.to(device, dtype=torch.float)

            outputs_class = model(input_ids,attention_masks,img,kg1,kg2,kg_sim,labels_img,args)
            outputs_class = outputs_class.squeeze()

            loss_class = criterion_clf(outputs_class,labels)

            loss = loss_class + args.lambda_orth*criterion_orth(p_img=model.module.ln_uniq_img.weight,\
                                                   p_txt=model.module.ln_uniq_txt.weight,\
                                                   w_shr=model.module.ln_shr.weight)
            if epoch < 5:
                optimizier.zero_grad()
                loss.backward()
                optimizier.step()
            else:
                optimizier2.zero_grad()
                loss.backward()
                optimizier2.step()
            correct = evaluation(outputs_class,labels)
            total_acc += (correct/args.batch)
            total_loss += loss.item()
            print('the running process is [{}]'.format(p),'[Epoch{}]'.format(epoch+1),\
                  "{}/{}".format(i+1,t_batch),"loss:",loss.item(),'loss_class:',loss_class.item(),"acc:",correct*100/args.batch)
        print('\nTrain | Loss:{:.5f} ACC:{:.3f}'.format(total_loss/t_batch, total_acc*100/t_batch))

        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0,0
            for i, (input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(val_loader):
                input_ids = input_ids.to(device, dtype=torch.long)
                attention_masks = attention_masks.to(device, dtype=torch.long)
                img = img.to(device)
                kg1 = kg1.to(device, dtype=torch.float)
                kg2 = kg2.to(device, dtype=torch.float)
                kg_sim = kg_sim.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                labels_img = labels_img.to(device, dtype=torch.float)
                outputs_class = model(input_ids, attention_masks, img, kg1, kg2, kg_sim,labels_img,args)
                outputs_class = outputs_class.squeeze()

                loss_class = criterion_clf(outputs_class, labels)

                loss = loss_class + args.lambda_orth * criterion_orth(p_img=model.module.ln_uniq_img.weight, \
                                                                      p_txt=model.module.ln_uniq_txt.weight, \
                                                                      w_shr=model.module.ln_shr.weight)
                correct = evaluation(outputs_class, labels)
                total_acc += (correct/args.batch)
                total_loss += loss.item()
            print("valid | Loss:{:.5f} ACC:{:.3f}".format(total_loss/v_batch, total_acc*100/v_batch))
            val_loss = total_loss/v_batch
            #------------------------earlystopping------------------------------
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                earlystopping.save_checkpoint(total_loss/v_batch, model)
        model.train()
        if epoch < 5:
            scheduler.step(val_loss)
        else:
            scheduler2.step(val_loss)
    model = earlystopping.load_model()
    test_result_dict = test_replace(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p)
    return test_result_dict

def test_replace(test_dict,model,device,config,dataset,model_mode,args,transform,pathset,p):
    test_result_dict = {}
    test_x_input_ids, test_x_attention_masks, test_x_img, \
    test_x_kg_1, test_x_kg_2, test_x_kg_sim, \
    test_y, test_y_img = test_dict['input_ids'], test_dict['attention_masks'], test_dict['image'], \
                       test_dict['vec_1'], test_dict['vec_2'], test_dict['sim_list'], \
                       test_dict['y'], test_dict['y_img']

    if args.lower == False:
        test_dataset = Dataset_all(x_txt_input_ids=test_x_input_ids, \
                                              x_txt_attention_masks=test_x_attention_masks, \
                                              x_img=test_x_img, x_kg1=test_x_kg_1, x_kg2=test_x_kg_2, \
                                              x_kg_sim=test_x_kg_sim, y=test_y, y_img=test_y_img, transform=transform,
                                              pathset=pathset)
    elif args.lower == True:
        test_dataset = Dataset_lower(x_txt_input_ids=test_x_input_ids, x_txt_attention_masks=test_x_attention_masks,
                                     x_img=test_x_img, x_kg1=test_x_kg_1, x_kg2=test_x_kg_2,x_kg_sim=test_x_kg_sim,
                                     y=test_y, y_img=test_y_img, transform=transform,pathset=pathset)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=args.batch,
                                             shuffle=False,
                                             # sampler=train_sampler,
                                             num_workers=2)
    model.eval()
    with torch.no_grad():
        for i, (input_ids, attention_masks, img, kg1, kg2, kg_sim, labels, labels_img) in enumerate(test_loader):
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_masks = attention_masks.to(device, dtype=torch.long)
            img = img.to(device)
            kg1 = kg1.to(device, dtype=torch.float)
            kg2 = kg2.to(device, dtype=torch.float)
            kg_sim = kg_sim.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            labels_img = labels_img.to(device, dtype=torch.float)
            outputs_class = model(input_ids, attention_masks, img, kg1, kg2, kg_sim,labels_img,args)
            outputs_class = outputs_class.squeeze()

            outputs_class[outputs_class >= 0.5] = 1
            outputs_class[outputs_class < 0.5] = 0
            print('output_class:',outputs_class)
            print('output_class size:', outputs_class.size())

            if i == 0:
                outputs_class_all = outputs_class
                labels_all = labels
            else:
                #TODO： 原始版本 但missing_patterns会报错
                # outputs_class_all = torch.cat([outputs_class_all,outputs_class],dim=0)
                # labels_all = torch.cat([labels_all,labels],dim=0)
                try:
                    outputs_class_all = torch.cat([outputs_class_all, outputs_class], dim=0)
                    labels_all = torch.cat([labels_all, labels], dim=0)
                except RuntimeError:
                    continue

    acc = accuracy(outputs_class_all,labels_all)
    f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake = macro_f1(
        outputs_class_all, labels_all, num_classes=2)
    print('----------------------------------------')
    print('acc:',acc,'prec:',precision,'rec:',recall,'f1:',f1)
    print('prec-fake:', precision_fake, 'rec-fake:', recall_fake, 'f1-fake:', f1_fake)
    print('prec-real:', precision_real, 'rec-real:', recall_real, 'f1-real:', f1_real)


    test_result_dict['acc'] = acc
    test_result_dict['prec'] = precision
    test_result_dict['rec'] = recall
    test_result_dict['f1'] = f1

    # test_dict['acc'] = acc
    test_result_dict['prec_fake'] = precision_fake
    test_result_dict['rec_fake'] = recall_fake
    test_result_dict['f1_fake'] = f1_fake

    test_result_dict['prec_real'] = precision_real
    test_result_dict['rec_real'] = recall_real
    test_result_dict['f1_real'] = f1_real

    with open('../result_missing_modality/result_test_{}_{}_batch{}_lr{}_orth{}_modal{}_{}.txt'.format(dataset,model_mode,args.batch, args.lr, args.lambda_orth, args.lambda_m,p),'w',encoding='utf-8',newline='')as f:
        string1 = 'acc:'+str(acc)+'\t'+'f1:'+str(f1)+'\t'+'precision:'+str(precision)+'\t'+'recall:'+str(recall)+'\n'
        string2 = 'f1_real:' + str(f1_real) + '\t' + 'precision_real:' + str(
            precision_real) + '\t' + 'recall_real:' + str(recall_real)+'\n'
        string3 = 'f1_fake:' + str(f1_fake) + '\t' + 'precision_fake:' + str(
            precision_fake) + '\t' + 'recall_fake:' + str(recall_fake)+'\n\n\n'
        f.writelines(string1)
        f.writelines(string2)
        f.writelines(string3)

    return test_result_dict

def main_twitter(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    model_mode = args.model
    config = inconsistency_Config()
    pathset = path_set_BERT(dataset)

    print("loading data...")
    if args.full_image == False:
        X_txt_train, X_img_train, X_kg_train, y_train, y_img_train,\
        X_txt_test, X_img_test, X_kg_test, y_test, y_img_test= pre_training_en(pathset, config)
    elif args.full_image == True:
        X_txt_train, X_img_train, X_kg_train, y_train, y_img_train, \
        X_txt_test, X_img_test, X_kg_test, y_test, y_img_test = pre_training_en_full_image(pathset, config)

    # 数据预处理部分，将数据分为train/valid/test并处理
    if args.full_image == True:
        preprocess = data_preprocess_ATT_bert_random_full_image(X_txt_train, X_img_train, X_kg_train, y_train, y_img_train, \
                                                     X_txt_test, X_img_test, X_kg_test, y_test, y_img_test, config,
                                                     pathset, args)
    else:
        preprocess = data_preprocess_ATT_bert_random(X_txt_train, X_img_train, X_kg_train, y_train, y_img_train,\
        X_txt_test, X_img_test, X_kg_test, y_test, y_img_test, config, pathset,args)

    train_dict_0, val_dict_0, test_dict_0, \
    train_dict_1, val_dict_1, test_dict_1, \
    train_dict_2, val_dict_2, test_dict_2, \
    train_dict_3, val_dict_3, test_dict_3, \
    train_dict_4, val_dict_4, test_dict_4 = preprocess.load_data()
    transform = preprocess.img_trans()


    # ------train&test----------------------------
    if model_mode == 'adversarial':
        test_result_dict_0 = train(train_dict_0, val_dict_0, test_dict_0, model_mode, device, config, dataset, args,
                                   transform, pathset, p=0)
        test_result_dict_1 = train(train_dict_1, val_dict_1, test_dict_1, model_mode, device, config, dataset,
                                   args, transform, pathset, p=1)
        test_result_dict_2 = train(train_dict_2, val_dict_2, test_dict_2, model_mode, device, config, dataset,
                                   args, transform, pathset, p=2)
        test_result_dict_3 = train(train_dict_3, val_dict_3, test_dict_3, model_mode, device, config, dataset,
                                   args, transform, pathset, p=3)
        test_result_dict_4 = train(train_dict_4, val_dict_4, test_dict_4, model_mode, device, config, dataset,
                                   args, transform, pathset, p=4)

    if model_mode == 'adversarial_step':
        test_result_dict_0 = train_step(train_dict_0, val_dict_0, test_dict_0, model_mode, device, config, dataset, args,
                                   transform, pathset, p=0)
        test_result_dict_1 = train_step(train_dict_1, val_dict_1, test_dict_1, model_mode, device, config, dataset,
                                   args, transform, pathset, p=1)
        test_result_dict_2 = train_step(train_dict_2, val_dict_2, test_dict_2, model_mode, device, config, dataset,
                                   args, transform, pathset, p=2)
        test_result_dict_3 = train_step(train_dict_3, val_dict_3, test_dict_3, model_mode, device, config, dataset,
                                   args, transform, pathset, p=3)
        test_result_dict_4 = train_step(train_dict_4, val_dict_4, test_dict_4, model_mode, device, config, dataset,
                                   args, transform, pathset, p=4)

    if model_mode == 'kdin' or 'add_token':
        test_result_dict_0 = train_replace(train_dict_0, val_dict_0, test_dict_0, model_mode, device, config, dataset, args,
                                   transform, pathset, p=0)
        test_result_dict_1 = train_replace(train_dict_1, val_dict_1, test_dict_1, model_mode, device, config, dataset,
                                   args, transform, pathset, p=1)
        test_result_dict_2 = train_replace(train_dict_2, val_dict_2, test_dict_2, model_mode, device, config, dataset,
                                   args, transform, pathset, p=2)
        test_result_dict_3 = train_replace(train_dict_3, val_dict_3, test_dict_3, model_mode, device, config, dataset,
                                   args, transform, pathset, p=3)
        test_result_dict_4 = train_replace(train_dict_4, val_dict_4, test_dict_4, model_mode, device, config, dataset,
                                   args, transform, pathset, p=4)

    # if model_mode == 'replace_step' or 'add_token_step':
    #     test_result_dict_0 = train_replace_step(train_dict_0, val_dict_0, test_dict_0, model_mode, device, config, dataset, args,
    #                                transform, pathset, p=0)
    #     test_result_dict_1 = train_replace_step(train_dict_1, val_dict_1, test_dict_1, model_mode, device, config, dataset,
    #                                args, transform, pathset, p=1)
    #     test_result_dict_2 = train_replace_step(train_dict_2, val_dict_2, test_dict_2, model_mode, device, config, dataset,
    #                                args, transform, pathset, p=2)
    #     test_result_dict_3 = train_replace_step(train_dict_3, val_dict_3, test_dict_3, model_mode, device, config, dataset,
    #                                args, transform, pathset, p=3)
    #     test_result_dict_4 = train_replace_step(train_dict_4, val_dict_4, test_dict_4, model_mode, device, config, dataset,
    #                                args, transform, pathset, p=4)


    dict_list = [test_result_dict_0, test_result_dict_1, test_result_dict_2, test_result_dict_3, test_result_dict_4]
    best_test_result_dict = {}
    best_test_result_dict['acc'] = 0
    best_test_result_dict['prec'] = 0
    best_test_result_dict['rec'] = 0
    best_test_result_dict['f1'] = 0
    for dic in dict_list:
        if dic['acc'] > best_test_result_dict['acc']:
            best_test_result_dict['acc'] = dic['acc']
            best_test_result_dict['prec'] = dic['prec']
            best_test_result_dict['rec'] = dic['rec']
            best_test_result_dict['f1'] = dic['f1']

    avg_test_result_dict = {}
    avg_test_result_dict['acc'] = (test_result_dict_0['acc'] + test_result_dict_1['acc'] + test_result_dict_2['acc'] +
                                   test_result_dict_3['acc'] \
                                   + test_result_dict_4['acc']) / 5
    avg_test_result_dict['prec'] = (test_result_dict_0['prec'] + test_result_dict_1['prec'] + test_result_dict_2['prec'] \
                                    + test_result_dict_3['prec'] + test_result_dict_4['prec']) / 5
    avg_test_result_dict['rec'] = (test_result_dict_0['rec'] + test_result_dict_1['rec'] + test_result_dict_2['rec'] \
                                   + test_result_dict_3['rec'] + test_result_dict_4['rec']) / 5
    avg_test_result_dict['f1'] = (test_result_dict_0['f1'] + test_result_dict_1['f1'] + test_result_dict_2['f1'] \
                                  + test_result_dict_3['f1'] + test_result_dict_4['f1']) / 5

    print('---------------------final best-------------------------------')
    print('parameter:', 'batch_size:', args.batch, 'epoch_num:', args.epoch, 'learning_rate:', args.lr)
    print('acc:', best_test_result_dict['acc'], 'prec:', best_test_result_dict['prec'], \
          'rec:', best_test_result_dict['rec'], 'f1:', best_test_result_dict['f1'])

    print('---------------------final average-------------------------------')
    print('parameter:', 'batch_size:', args.batch, 'epoch_num:', args.epoch, 'learning_rate:', args.lr)
    print('acc:', avg_test_result_dict['acc'], 'prec:', avg_test_result_dict['prec'], \
          'rec:', avg_test_result_dict['rec'], 'f1:', avg_test_result_dict['f1'])

    print('finish')

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    model_mode = args.model
    config = inconsistency_Config()
    pathset = path_set_BERT(dataset)

    print("loading data...")
    if dataset == 'pheme' and args.full_image == False:
        X_txt, X_img, X_kg, y, y_img = pre_training_pheme(pathset, config)
    elif dataset == 'pheme' and args.full_image == True:
        X_txt, X_img, X_kg, y, y_img = pre_training_pheme_full_image(pathset, config)

    elif dataset == 'weibo' and args.full_image == True:
        X_txt, X_img, X_kg, y, y_img = pre_training_zh_full_image(pathset, config)
    else:
        X_txt, X_img, X_kg, y, y_img = pre_training_zh(pathset,config)

    #数据预处理部分，将数据分为train/valid/test并处理
    if args.full_image == True:
        preprocess = data_preprocess_ATT_bert_nfold_random_full_image(X_txt, X_img, X_kg, y, y_img, config, pathset, args)
    else:
        preprocess = data_preprocess_ATT_bert_nfold_random(X_txt, X_img,X_kg,y,y_img,config,pathset,args)

    train_dict_0, val_dict_0, test_dict_0, \
    train_dict_1, val_dict_1, test_dict_1, \
    train_dict_2, val_dict_2, test_dict_2, \
    train_dict_3, val_dict_3, test_dict_3, \
    train_dict_4, val_dict_4, test_dict_4 = preprocess.load_data()
    transform = preprocess.img_trans()

    #model类型声明要放在后面，不然就是对一个模型参数进行改进
    # if model_mode == 'initial':
    #     model = inconsistency_model_initial(config=config,pathset=pathset)
    # elif model_mode == 'weight':
    #     model = inconsistency_model_weight(config=config, pathset=pathset)
    # elif model_mode == 'adversial':
    #     model = inconsistency_model_adversial(config=config,pathset=pathset)
    # # elif model_mode == 'original':
    # #     pass
    # model = model.to(device)
    # if torch.cuda.device_count()>1:
    #     model = torch.nn.DataParallel(model)

    #------train&test----------------------------
    # 输出包含modal_loss
    if model_mode == 'adversarial':
        test_result_dict_0 = train(train_dict_0,val_dict_0,test_dict_0, model_mode,device,config,dataset,args,transform,pathset,p=0)
        test_result_dict_1 = train(train_dict_1, val_dict_1, test_dict_1, model_mode, device, config, dataset,
                                        args,transform,pathset,p=1)
        test_result_dict_2 = train(train_dict_2, val_dict_2, test_dict_2, model_mode, device, config, dataset,
                                        args,transform,pathset,p=2)
        test_result_dict_3 = train(train_dict_3, val_dict_3, test_dict_3, model_mode, device, config, dataset,
                                        args,transform,pathset,p=3)
        test_result_dict_4 = train(train_dict_4, val_dict_4, test_dict_4, model_mode, device, config, dataset,
                                        args,transform,pathset,p=4)
    #分步训练
    if model_mode == 'adversarial_step':
        test_result_dict_0 = train_step(train_dict_0,val_dict_0,test_dict_0, model_mode,device,config,dataset,args,transform,pathset,p=0)
        test_result_dict_1 = train_step(train_dict_1, val_dict_1, test_dict_1, model_mode, device, config, dataset,
                                        args,transform,pathset,p=1)
        test_result_dict_2 = train_step(train_dict_2, val_dict_2, test_dict_2, model_mode, device, config, dataset,
                                        args,transform,pathset,p=2)
        test_result_dict_3 = train_step(train_dict_3, val_dict_3, test_dict_3, model_mode, device, config, dataset,
                                        args,transform,pathset,p=3)
        test_result_dict_4 = train_step(train_dict_4, val_dict_4, test_dict_4, model_mode, device, config, dataset,
                                        args,transform,pathset,p=4)

    # 输入包含img_label和输出不包含modal_loss
    if model_mode == 'kdin' or 'add_token':
        test_result_dict_0 = train_replace(train_dict_0, val_dict_0, test_dict_0, model_mode, device, config,dataset,
                                           args, transform, pathset, p=0)
        test_result_dict_1 = train_replace(train_dict_1, val_dict_1, test_dict_1, model_mode, device, config,dataset,
                                           args, transform, pathset, p=1)
        test_result_dict_2 = train_replace(train_dict_2, val_dict_2, test_dict_2, model_mode, device, config,dataset,
                                           args, transform, pathset, p=2)
        test_result_dict_3 = train_replace(train_dict_3, val_dict_3, test_dict_3, model_mode, device, config,dataset,
                                           args, transform, pathset, p=3)
        test_result_dict_4 = train_replace(train_dict_4, val_dict_4, test_dict_4, model_mode, device, config,dataset,
                                           args, transform, pathset, p=4)

    # if model_mode == 'replace_step' or 'add_token_step':
    #     test_result_dict_0 = train_replace_step(train_dict_0, val_dict_0, test_dict_0, model_mode, device, config,dataset,
    #                                        args, transform, pathset, p=0)
    #     test_result_dict_1 = train_replace_step(train_dict_1, val_dict_1, test_dict_1, model_mode, device, config,dataset,
    #                                        args, transform, pathset, p=1)
    #     test_result_dict_2 = train_replace_step(train_dict_2, val_dict_2, test_dict_2, model_mode, device, config,dataset,
    #                                        args, transform, pathset, p=2)
    #     test_result_dict_3 = train_replace_step(train_dict_3, val_dict_3, test_dict_3, model_mode, device, config,dataset,
    #                                        args, transform, pathset, p=3)
    #     test_result_dict_4 = train_replace_step(train_dict_4, val_dict_4, test_dict_4, model_mode, device, config,dataset,
    #                                        args, transform, pathset, p=4)

    #输入包含img_label和输出包含modal_loss
    if model_mode == 'initial':
        test_result_dict_0 = train_initial(train_dict_0, val_dict_0, test_dict_0, model_mode, device, config, dataset, args,
                                   transform, pathset, p=0)
        test_result_dict_1 = train_initial(train_dict_1, val_dict_1, test_dict_1, model_mode, device, config, dataset,
                                   args, transform, pathset, p=1)
        test_result_dict_2 = train_initial(train_dict_2, val_dict_2, test_dict_2, model_mode, device, config, dataset,
                                   args, transform, pathset, p=2)
        test_result_dict_3 = train_initial(train_dict_3, val_dict_3, test_dict_3, model_mode, device, config, dataset,
                                   args, transform, pathset, p=3)
        test_result_dict_4 = train_initial(train_dict_4, val_dict_4, test_dict_4, model_mode, device, config, dataset,
                                   args, transform, pathset, p=4)

    if model_mode == 'initial_step':
        test_result_dict_0 = train_initial_step(train_dict_0, val_dict_0, test_dict_0, model_mode, device, config, dataset, args,
                                   transform, pathset, p=0)
        test_result_dict_1 = train_initial_step(train_dict_1, val_dict_1, test_dict_1, model_mode, device, config, dataset,
                                   args, transform, pathset, p=1)
        test_result_dict_2 = train_initial_step(train_dict_2, val_dict_2, test_dict_2, model_mode, device, config, dataset,
                                   args, transform, pathset, p=2)
        test_result_dict_3 = train_initial_step(train_dict_3, val_dict_3, test_dict_3, model_mode, device, config, dataset,
                                   args, transform, pathset, p=3)
        test_result_dict_4 = train_initial_step(train_dict_4, val_dict_4, test_dict_4, model_mode, device, config, dataset,
                                   args, transform, pathset, p=4)

    dict_list = [test_result_dict_0,test_result_dict_1,test_result_dict_2,test_result_dict_3,test_result_dict_4]
    best_test_result_dict = {}
    best_test_result_dict['acc'] = 0
    best_test_result_dict['prec'] = 0
    best_test_result_dict['rec'] = 0
    best_test_result_dict['f1'] = 0
    for dic in dict_list:
        if dic['acc'] > best_test_result_dict['acc']:
            best_test_result_dict['acc'] = dic['acc']
            best_test_result_dict['prec'] = dic['prec']
            best_test_result_dict['rec'] = dic['rec']
            best_test_result_dict['f1'] = dic['f1']

    avg_test_result_dict = {}
    avg_test_result_dict['acc'] = (test_result_dict_0['acc']+test_result_dict_1['acc']+test_result_dict_2['acc']+test_result_dict_3['acc']\
                                    +test_result_dict_4['acc'])/5
    avg_test_result_dict['prec'] = (test_result_dict_0['prec'] + test_result_dict_1['prec'] + test_result_dict_2['prec'] \
                                    + test_result_dict_3['prec'] + test_result_dict_4['prec']) / 5
    avg_test_result_dict['rec'] = (test_result_dict_0['rec'] + test_result_dict_1['rec'] + test_result_dict_2['rec'] \
                                    + test_result_dict_3['rec'] + test_result_dict_4['rec']) / 5
    avg_test_result_dict['f1'] = (test_result_dict_0['f1'] + test_result_dict_1['f1'] + test_result_dict_2['f1'] \
                                    + test_result_dict_3['f1'] + test_result_dict_4['f1']) / 5


    print('---------------------final best-------------------------------')
    print('parameter:', 'batch_size:', args.batch, 'epoch_num:', args.epoch, 'learning_rate:', args.lr)
    print('acc:', best_test_result_dict['acc'], 'prec:', best_test_result_dict['prec'], \
          'rec:', best_test_result_dict['rec'], 'f1:', best_test_result_dict['f1'])

    print('---------------------final average-------------------------------')
    print('parameter:', 'batch_size:', args.batch, 'epoch_num:', args.epoch, 'learning_rate:', args.lr)
    print('acc:', avg_test_result_dict['acc'], 'prec:', avg_test_result_dict['prec'], \
          'rec:', avg_test_result_dict['rec'], 'f1:', avg_test_result_dict['f1'])

    print('finish')

if __name__ == "__main__":
    #------------------------------------------------------------------------
    #随机数种子
    # config = inconsistency_Config()
    # seed = config.seed
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    #
    # np.random.seed(seed)
    # random.seed(seed)
    #------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='inconsistency_matters')
    parser.add_argument('--dataset',type=str,default='pheme')#twitter/weibo
    parser.add_argument('--model',type=str,default='adversarial')#initial调整初始化/weight调整权重/adversarial对抗学习/original(emnlp模型)/replace 用文本T_u和T_s代替I_u和I_s/add_token增加标志位
    parser.add_argument('--cuda',type=str,default='0')
    parser.add_argument('--epoch',type=int,default=50)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--lambda_orth',type=float,default=1.5)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--lambda_m',type=float,default=1.0)
    # parser.add_argument('--seed',type=int,default=1)
    #增加参数 false指使用目前图片缺失数据集，true在后续data_load中不同
    parser.add_argument('--lower',type=bool,default=False)
    #增加参数full_image false指图片缺失数据集，true指图片不缺失的小数据集,在util部分不同
    parser.add_argument('--full_image',type=bool,default=False)
    args = parser.parse_args()
    if args.dataset == "twitter":
        main_twitter(args)
    else:
        main(args)

