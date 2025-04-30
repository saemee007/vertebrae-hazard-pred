import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def save_result(args):

    train_tot = pd.DataFrame()
    val_tot = pd.DataFrame()
    os.makedirs(f'{args.output}/vis', exist_ok=True)
    
    with open(f'{args.output}/result_train.csv', 'r') as f:
        for i, epoch_metrics in enumerate([line.split(',')[:-1] for line in f.readlines()]):
            for j, epoch_metric in enumerate(epoch_metrics):
                if j % 2 == 0:
                    train_tot.loc[i,epoch_metric] = np.round(float(epoch_metrics[j+1]), 2)
                else:
                    continue
    
    with open(f'{args.output}/result_val.csv', 'r') as f:
        for i, epoch_metrics in enumerate([line.split(',')[:-1] for line in f.readlines()]):
            for j, epoch_metric in enumerate(epoch_metrics):
                if j % 2 == 0:
                    val_tot.loc[i,epoch_metric] = np.round(float(epoch_metrics[j+1]), 2)
                else:
                    continue
            
    if args.pred_enabled:
        plt.clf()
        plt.plot(list(train_tot['epoch']), list(train_tot[f'PRED_c_index']), label='train')
        plt.plot(list(val_tot['epoch']), list(val_tot[f'PRED_c_index']), label='valid')
        plt.title(f'PRED_c_index')
        plt.legend()
        plt.savefig(f'{args.output}/vis/PRED_c_index.png')

    if args.cls_enabled:
        plt.clf()
        plt.plot(list(train_tot['epoch']), list(train_tot['CLS_AUC']), label='train')
        plt.plot(list(val_tot['epoch']), list(val_tot['CLS_AUC']), label='valid')
        plt.title('CLS AUC')
        plt.legend()
        plt.savefig(f'{args.output}/vis/CLS_AUC.png')
