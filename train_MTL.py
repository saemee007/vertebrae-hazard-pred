import os
import torch
import torch.optim as optim

from utils.loss import MultiTaskLoss
from utils.runner import run
from utils.engine import evaluate
from utils.parsing import parse_argument, set_seed
from utils.optimizer import WeightMethods, extract_weight_method_parameters_from_args

from network.model import MultiTaskLearner
from data import get_dataloader

import warnings

warnings.filterwarnings(action='ignore')

def main():

    # Option
    args = parse_argument()

    # GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    os.environ["OMP_NUM_THREADS"] = "2"
    
    # Dataloader
    cls_train_loader, cls_val_loader, pred_train_loader, pred_val_loader, args = get_dataloader(args)
    
    # Model 
    model = MultiTaskLearner(args)
    model.to(args.device)

    # Loss Function
    criterion = MultiTaskLoss(args) 

    if (args.task == 'fine-tuning') & (args.mode == 'train'):
        
        print(f'\nLoad weight from "{args.weight}/checkpoint_best.pth"\n')
        state_dict = torch.load(os.path.join(args.weight,'checkpoint_best.pth'))['state_dict']
        pretrained_dict = {k: v for k, v in state_dict.items() if k.startswith('backbone')}
        model.encoder.load_state_dict(pretrained_dict)
        
        lr = float(args.lr) / 10
        
    else:
        if args.weight is not None:
            print(f'\nLoad weight from "{args.weight}/checkpoint_best.pth"\n')
            obj = torch.load(os.path.join(args.weight,'checkpoint_best.pth'))
            model.load_state_dict(obj['state_dict'])
            args.start_epoch = obj['epoch']
            
        lr = float(args.lr)

    if args.mode == 'train':
        length = min((len(pred_train_loader), len(cls_train_loader)))
    elif args.mode == 'test':
        length = min((len(pred_val_loader), len(cls_val_loader)))
                
    # Optimizer
    if args.nashmtl:
        weight_methods_parameters = extract_weight_method_parameters_from_args()
        method = WeightMethods(method='nashmtl', n_tasks=2, device=args.device, **weight_methods_parameters['nashmtl'])
        optimizer = optim.Adam(
            [
                dict(params=model.parameters(), lr=lr),
                dict(params=method.parameters(), lr=0.025)
            ],
                                )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    else:
        method = None
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=length, epochs=args.nepochs)
    
    if args.mode == 'train':
        run(args, cls_train_loader, cls_val_loader, pred_train_loader, pred_val_loader, model, criterion, optimizer, scheduler, method)

    elif (args.mode == 'test'):
        evaluate(args, model, cls_val_loader, pred_val_loader)

if __name__ == '__main__':
    main()