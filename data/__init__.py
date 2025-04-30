import torch
import torchio as tio
from utils.parsing import set_seed
from utils.metrics import get_censoring_dist
from data.dataset import CLSDataset, PREDDataset

def get_dataloader(args):
    #set_seed()

    cls_train_loader = None
    pred_train_loader = None

    val_augmentator = tio.Compose([
        tio.Resize((32, 64, 112))
    ], p=1.0)
    
    if args.task == 'MTL':
        cls_val_dset = CLSDataset(
            args,
            transform=val_augmentator, 
            subset='val')
    else: 
        cls_val_dset = CLSDataset(
        args,
        transform=val_augmentator, 
        subset='val')
        
    pred_val_dset = PREDDataset(
        args,
        transform=val_augmentator, 
        subset='val')

    if args.pred_enabled:
        args.censoring_distribution = {
            '1': 0.9417989417989415, 
            '2': 0.8816841157266683, 
            '3': 0.8204560521345382, 
            '4': 0.7688189579442527, 
            '5': 0.6861502527889567, 
            '6': 0.6188806201625885, 
            '7': 0.5121770649621422, 
            '8': 0.4438867896338566, 
            '9': 0.4438867896338566} 
        # get_censoring_dist(pred_val_dset)

    cls_val_loader = torch.utils.data.DataLoader(
        cls_val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, 
        drop_last=False)
    
    pred_val_loader = torch.utils.data.DataLoader(
        pred_val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, 
        drop_last=False)
    
    print(f'Loaded Classification Validation Dataset ... ({len(cls_val_dset)})')
    print(f'Loaded Prediction Validation Dataset ... ({len(pred_val_dset)})')
    print()
    
    if args.mode == 'train':
        train_augmentator = tio.Compose([
            tio.RandomAffine(scales=(0.8, 1.20), degrees=(-5,5,0,0,0,0,)),
            tio.RandomFlip(axes=('LR')),
            tio.Resize((32, 64, 112))
        ], p=1.0)
        
        cls_train_dset = CLSDataset(
            args,
            transform=train_augmentator, 
            subset='train')

        pred_train_dset = PREDDataset(
            args,
            transform=train_augmentator, 
            subset='train')
        
        cls_train_loader = torch.utils.data.DataLoader(
            cls_train_dset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, 
            drop_last=True) 

        pred_train_loader = torch.utils.data.DataLoader(
            pred_train_dset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, 
            drop_last=True) 

        print(f'Loaded Classification Train Dataset ... ({len(cls_train_dset)})')
        print(f'Loaded Prediction Train Dataset ... ({len(pred_train_dset)})')
        print()
        
    return cls_train_loader, cls_val_loader, pred_train_loader, pred_val_loader, args

