import os
import itertools
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix

from utils import log
from utils import metrics
from utils.parsing import set_seed

def train(args, cls_train_loader, pred_train_loader, model, criterion, \
        optimizer, scaler, scheduler, method):
    
    model.train()
    
    cls_sample_list = []
    pred_sample_list = []
    cls_output_list = []
    pred_output_list = [] 
    
    running_loss = 0.
    tot_len = 0
    
    for i, (cls_samples, pred_samples) in enumerate(zip(cls_train_loader, pred_train_loader)):
        
        optimizer.zero_grad()
        
        if args.cls_enabled:
            b = cls_samples['vers_img_tensor'].size(0)
            cls_samples = {k: v.to(args.device) for k, v in cls_samples.items() if torch.is_tensor(v)}
            cls_sample_list.append(cls_samples)
            
        if args.pred_enabled:
            b = pred_samples['vers_img_tensor'].size(0)
            pred_samples = {k: v.to(args.device) for k, v in pred_samples.items() if torch.is_tensor(v)}
            pred_sample_list.append(pred_samples)
            
            
        with torch.cuda.amp.autocast():
            cls_output, pred_output, features = model(cls_samples, pred_samples) 
            cls_output_list.append(cls_output['prob'])
            pred_output_list.append(pred_output['prob'])
            
            weighted_cls_loss, weighted_pred_loss = criterion(cls_output['prob'], pred_output['logit'], cls_samples, pred_samples)
    
        if args.task == 'MTL':
            losses = torch.stack([weighted_cls_loss, weighted_pred_loss])
        else:
            if args.cls_enabled:
                losses = [weighted_cls_loss]
            if args.pred_enabled:
                losses = [weighted_pred_loss]
        
        if args.nashmtl:
            _ = method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
                ) 
            optimizer.step()   
        else:
            scaler.scale(sum(losses)).backward()
            scaler.step(optimizer)
            scaler.update()
            
        running_loss += sum(losses) * b
        tot_len += b
    scheduler.step()
    
    cls_metrics_dict, pred_metrics_dict = {}, {}
    if args.cls_enabled:
        cls_metrics_dict = compute_cls_epoch_metrics(args, cls_sample_list, cls_output_list)

    if args.pred_enabled:
        pred_metrics_dict = compute_pred_epoch_metrics(args, pred_sample_list, pred_output_list)
    
    return running_loss/tot_len, cls_metrics_dict, pred_metrics_dict

def inference(args, cls_val_loader, pred_val_loader, model, criterion=None):

    model.eval()
    
    cls_metrics_dict, pred_metrics_dict = {}, {}
    with torch.no_grad():
        if args.cls_enabled:
            cls_sample_list = []
            mean_cls_pred_proba_list = []
            for cls_samples in cls_val_loader:     
                cls_pred_proba_list = []           
                cls_samples = {k: v.to(args.device) for k, v in cls_samples.items() if torch.is_tensor(v)}
                cls_sample_list.append(cls_samples)
                
                # Inference
                if args.pred_enabled:
                    cls_output, _, _ = model(cls_samples, cls_samples, turn_off = ['pred'])
                else:
                    cls_output, _, _ = model(cls_samples, cls_samples)
                cls_pred_proba_list.append(cls_output['prob'])
                
                # Ensemble
                for _ in range(args.ensemble - 1):
                    if args.pred_enabled:
                        cls_output, _, _ = model(cls_samples, cls_samples, turn_off = ['pred'])
                    else:
                        cls_output, _, _ = model(cls_samples, cls_samples)
                        
                    cls_pred_proba_list.append(cls_output['prob'])   
                mean_cls_pred_proba_arr = torch.mean(torch.stack(cls_pred_proba_list), 0)
                mean_cls_pred_proba_list.append(mean_cls_pred_proba_arr)
                
            cls_metrics_dict = compute_cls_epoch_metrics(args, cls_sample_list, mean_cls_pred_proba_list)
        
        if args.pred_enabled:
            pred_sample_list = []
            pred_output_list = []  
            mean_pred_pred_proba_list = []
            for pred_samples in pred_val_loader:
                pred_pred_proba_list = []
                pred_samples = {k: v.to(args.device) for k, v in pred_samples.items() if torch.is_tensor(v)}
                pred_sample_list.append(pred_samples)
                
                # Inference
                if args.cls_enabled:
                    _, pred_output, _ = model(pred_samples, pred_samples, turn_off = ['cls'])
                else:
                    _, pred_output, _ = model(pred_samples, pred_samples)
                                        
                pred_output_list.append(pred_output['logit'])
                pred_pred_proba_list.append(pred_output['prob'])
                
                # Ensemble
                for _ in range(args.ensemble - 1):
                    if args.cls_enabled:
                        _, pred_output, _ = model(pred_samples, pred_samples, turn_off = ['cls'])
                    else:
                        _, pred_output, _ = model(pred_samples, pred_samples)
                                            
                    pred_pred_proba_list.append(pred_output['prob'])                    
                mean_pred_pred_proba_arr = torch.mean(torch.stack(pred_pred_proba_list), 0)
                mean_pred_pred_proba_list.append(mean_pred_pred_proba_arr)
            
            pred_metrics_dict = compute_pred_epoch_metrics(args, pred_sample_list, mean_pred_pred_proba_list)
            
        if criterion is not None:
            weighted_cls_loss, weighted_pred_loss = criterion(mean_cls_pred_proba_list, pred_output_list, cls_sample_list, pred_sample_list)
            
            if args.task == 'MTL':
                losses = [weighted_cls_loss, weighted_pred_loss]
            else:
                if args.cls_enabled:
                    losses = [weighted_cls_loss]
                if args.pred_enabled:
                    losses = [weighted_pred_loss]
            loss = sum(losses)
        else:
            loss = None
            
    return loss, cls_metrics_dict, pred_metrics_dict

def evaluate(args, model, cls_val_loader, pred_val_loader):
    
    if not os.path.isfile('record.txt'):
        writing_mode = 'w' # write
    else:
        writing_mode = 'a' # edit

    config_file = args.config.split('/')[-1].split('.')[0]
    write_content = f'{config_file}_{args.task}_{args.view}_{args.split}_{args.mode}_{args.slices_select}_{args.channels_num_2d}slices_{args.pred_data.split("/")[-1].split("_")[1]},'
    log.start_log('record.txt', writing_mode, write_content)
    
    if (args.mode == 'train') | (args.weight is None):
        print(f'\nLoad weight from "{args.output}/checkpoint_best.pth"\n')
        obj = torch.load(os.path.join(args.output,'checkpoint_best.pth'))
    else: 
        print(f'\nLoad weight from "{args.weight}"\n')
        obj = torch.load(args.weight)

    # TODO
    if (args.task == 'CLS') : # & (args.view == 'axial')
        state_dict = obj['state_dict']
        encoder_dict = {k: v for k, v in state_dict.items() if k.startswith('backbone')}
        decoder_dict = {k.replace('classifier.',''): v for k, v in state_dict.items() if k.startswith('classifier')}
        model.encoder.load_state_dict(encoder_dict)
        model.decoders.cls_decoder.load_state_dict(decoder_dict)

    else:
        model.load_state_dict(obj['state_dict'])
        
    _, cls_metrics_dict, pred_metrics_dict = inference(args, cls_val_loader, pred_val_loader, model)
    
    if args.cls_enabled:
        for k, v in cls_metrics_dict.items():
            print(f'CLS {k}: {v}')
        print()
    
    if args.pred_enabled:
        for k, v in pred_metrics_dict.items():
            print(f'PRED {k}: {v}')
        print()

    log.log_dict('record.txt', cls_metrics_dict, pred_metrics_dict)

def compute_cls_epoch_metrics(args, sample_list, proba_list):
    cls_gt_label_list = []
    cls_pred_label_list = []
    cls_pred_proba_list = []

    for (sample, cls_pred_proba) in zip(sample_list, proba_list):
        cls_pred_proba = cls_pred_proba.detach().cpu().numpy()
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.cpu().numpy()
        
        y = sample['y']
        cls_pred_label = (cls_pred_proba[:, -1] > 0.5).reshape(-1)
        cls_pred_proba = cls_pred_proba.reshape((-1, cls_pred_proba.shape[-1]))
        
        cls_gt_label_list.extend(y)
        cls_pred_label_list.extend(cls_pred_label)
        cls_pred_proba_list.extend(cls_pred_proba.tolist())
        
    metrics_dict = {}
    metrics_dict['accuracy'] = accuracy_score(cls_gt_label_list, cls_pred_label_list)
    metrics_dict['AUC'] = roc_auc_score(cls_gt_label_list, np.array(cls_pred_proba_list)[:, -1], multi_class='ovo')
    metrics_dict['F1'] = f1_score(cls_gt_label_list, cls_pred_label_list, average='weighted')
    
    cf = confusion_matrix(cls_gt_label_list, cls_pred_label_list)
    metrics_dict['sensitivity'] = cf[0,0]/(cf[0,0]+cf[0,1])
    metrics_dict['specificity'] = cf[1,1]/(cf[1,0]+cf[1,1])

    return metrics_dict    

def compute_pred_epoch_metrics(args, sample_list, output_list, metric_list=[metrics.get_survival_metrics]):
    pred_gt_label_list = []
    pred_censors_list = []
    pred_pred_proba_list = []
    for sample, output in zip(sample_list, output_list):
        proba = output.detach().cpu().numpy()
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.cpu().numpy()
        
        y = sample['y']
        censors = sample['time_at_event']
        
        pred_gt_label_list.extend(y)
        pred_censors_list.extend(censors)
        pred_pred_proba_list.extend(proba)
        
    logging_dict = {}
    logging_dict['y'] = np.array(pred_gt_label_list)
    logging_dict['probs'] = pred_pred_proba_list
    logging_dict['censors'] = np.array(pred_censors_list) 
    
    metrics_dict = {}
    for metric_func in metric_list:
        metric_dict = metric_func(args, logging_dict)
        for k, v in metric_dict.items():
            metrics_dict[k] = v
            
    return metrics_dict