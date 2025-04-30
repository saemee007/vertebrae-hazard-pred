import os

def start_log(file_name, writing_mode='w', write_content=''):
    fconv = open(file_name, writing_mode)
    fconv.write(write_content)
    fconv.close()

def log_dict(file_name, cls_metrics_dict, pred_metrics_dict, epoch=None, loss=None, sep=','):
    fconv = open(file_name, 'a')
    if epoch is not None:
        fconv.write(f'epoch{sep}{epoch+1}{sep}')
    
    for k, v in cls_metrics_dict.items():
        fconv.write(f'CLS_{k}{sep}{v}{sep}')
    
    for k, v in pred_metrics_dict.items():
        fconv.write(f'PRED_{k}{sep}{v}{sep}')
        
    fconv.write('\n')
    fconv.close()