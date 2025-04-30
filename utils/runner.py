import os
import torch
import numpy as np
from tqdm import tqdm

from utils.engine import train, inference, evaluate
from utils import log
from utils.visualize import save_result

def run(args, cls_train_loader, cls_val_loader, pred_train_loader, pred_val_loader, model, criterion, optimizer, scheduler, method):
    
    #open output file
    os.makedirs(args.output, exist_ok=True)
    if (not os.path.isfile(os.path.join(args.output,'result_train.csv'))) or (not os.path.join(args.output,'result_val.csv')):
        writing_mode = 'w' # write
    else:
        writing_mode = 'a' # edit

    log.start_log(os.path.join(args.output,'result_train.csv'), writing_mode)
    log.start_log(os.path.join(args.output,'result_val.csv'), writing_mode)
    
    scaler = torch.cuda.amp.GradScaler()

    #loop throuh epochs
    best_perfo = 0.
    best_auc = 0.

    for epoch in tqdm(np.arange(args.start_epoch, args.nepochs)):
        
        loss, cls_metrics_dict, pred_metrics_dict = train(args, cls_train_loader, pred_train_loader, model, criterion, optimizer, scaler, scheduler, method)
        
        if args.pred_enabled:
            print('Training\tEpoch: [{}/{}]\tLoss: {}\t C-index: {}'.format(epoch+1, args.nepochs, loss, pred_metrics_dict['c_index']))
        else:
            print('Training\tEpoch: [{}/{}]\tLoss: {}\t AUC: {}'.format(epoch+1, args.nepochs, loss, cls_metrics_dict['AUC']))
            
        log.log_dict(os.path.join(args.output, 'result_train.csv'), cls_metrics_dict, pred_metrics_dict, epoch, loss)

        #Validation
        if (epoch+1) % args.test_every == 0:

            loss, cls_metrics_dict, pred_metrics_dict = inference(args, cls_val_loader, pred_val_loader, model, criterion)
            
            if args.pred_enabled:
                print('Validation\tEpoch: [{}/{}]\tLoss: {}\t C-index: {}'.format(epoch+1, args.nepochs, loss,  pred_metrics_dict['c_index']))
            else:
                print('Training\tEpoch: [{}/{}]\tLoss: {}\t AUC: {}'.format(epoch+1, args.nepochs, loss, cls_metrics_dict['AUC']))
                
            log.log_dict(os.path.join(args.output, 'result_val.csv'), cls_metrics_dict, pred_metrics_dict, epoch, loss)
            save_result(args)
            
            # Select metric for saving model
            if args.saving_model_metric == 'cindex':
                saving_model_metric = pred_metrics_dict['c_index']
            elif args.saving_model_metric == 'cls_auc':
                saving_model_metric =  cls_metrics_dict['AUC']
                
            # Save best model
            if args.pred_enabled:
                if saving_model_metric >= best_perfo:
                    best_perfo = saving_model_metric
                    obj = {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'best_perfo': best_perfo,
                    }
                    torch.save(obj, os.path.join(args.output,'checkpoint_best.pth'))
                    
            else:
                if cls_metrics_dict['AUC'] >= best_perfo:
                    best_auc = cls_metrics_dict['AUC']
                    obj = {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'AUC': best_auc,
                    }
                    torch.save(obj, os.path.join(args.output,'checkpoint_best.pth'))
    
    evaluate(args, model, cls_val_loader, pred_val_loader)