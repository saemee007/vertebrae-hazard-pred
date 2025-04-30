import os
import h5py
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchio as tio
import torch.utils.data as data

class MultitaskDataset(data.Dataset):
    def __init__(self, args, transform=None, subset='train'):
        return
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        raise NotImplementedError
        

    def create_dataset(self, data, max_fracture_num=3):
        dataset = []
        
        for idx in data.index.to_list():
            sample = self.get_sample_dict(idx, data, max_fracture_num)
            if len(sample) == 0:
                continue
            
            dataset.append(sample)
        
        return dataset

    def get_sample_dict(self, idx, data, max_fracture_num):
        raise NotImplementedError
        
    def load_ver_img_list(self, ver_img_list, ver_path, args, transform, patch_size=(24, 128, 224), channels_num_2d=3):

        with h5py.File(ver_path, 'r') as f:
            img = np.array(f['image'])
            muscle_mask = np.array(f['muscle_mask'])
            verte_mask = np.array(f['vertebrae_mask'])

            muscle_mask[verte_mask == 1] = 0

            img = np.clip(img, 0, 1000)
            img = (img - img.min()) / (img.max() - img.min())
            
            if args.bone_only == 'bone_only':
                ver_img = img.copy()

                ver_img[verte_mask == 0] = 0
                bbox = self.get_roi_bbox(verte_mask, patch_size)
                ver_img = self.crop_roi(ver_img, bbox)

                if transform is not None:
                    subject = tio.Subject(image=tio.ScalarImage(tensor=ver_img[None]))
                    subject = transform.apply_transform(subject)
                    ver_img = subject.image.tensor.numpy().astype(np.float32)[0]
                resized_img = ver_img.copy()

            elif args.bone_only == 'bone_muscle':
                ver_img = img.copy()
                muscle_img = img.copy()
                
                ver_img[verte_mask == 0] = 0
                muscle_img[muscle_mask == 0] = 0

                if transform is not None:
                    subject = tio.Subject(
                        ver_img=tio.ScalarImage(tensor=ver_img[None]),
                        muscle_img=tio.ScalarImage(tensor=muscle_img[None]))
                    subject = transform.apply_transform(subject)
                    
                    ver_img = subject.ver_img.tensor.numpy().astype(np.float32)[0]
                    muscle_img = subject.muscle_img.tensor.numpy().astype(np.float32)[0]
                    resized_img = ver_img.copy() + muscle_img.copy()
        
        if args.view == 'sagittal':
            ver_img = np.transpose(ver_img, (2, 0, 1))[:,::-1]
            resized_img = np.transpose(resized_img, (2, 0, 1))[:,::-1]

        if args.dimension == '2D':
            if args.slices_select == 'random':
                if args.adjust_random == 'non_overlap':
                    patch_slice = np.where(ver_img.sum((1, 2)))[0]
                    space_num = len(patch_slice) - channels_num_2d * 3
                    
                    cases = list(range(channels_num_2d + space_num))
                    selected_cases = sorted(random.sample(cases, channels_num_2d))

                    slice_num = patch_slice.min()                
                    for i in cases:
                        if i in selected_cases:
                            ver_img_list.append(resized_img[slice_num:slice_num + 3])
                            slice_num += 3
                        else:
                            slice_num += 1
                            
                else:
                    patch_slice = sorted(np.where(ver_img.sum((1, 2)))[0])[1:-1]
                    probs = []
                    if args.adjust_random == 'gaussian':
                        from scipy.stats import norm

                        mean = patch_slice[int((len(patch_slice) - 1) / 2)]
                        var = 2 
                        probs = norm(mean, var).pdf(patch_slice) 
                        probs = probs / sum(probs)
                    elif args.adjust_random == 'overlap':
                        for i in range(len(patch_slice)):
                            probs.append(1 / len(patch_slice))                     
                        
                    try:
                        selected_cases = sorted(np.random.choice(patch_slice, channels_num_2d, replace=False, p=probs))
                        for i in selected_cases:
                            ver_img_list.append(resized_img[i - 1 : i + 2])
                    except:
                        print('The number of vertebrae slices is insufficient')
                        print(f'{ver_path}/n')
                        ver_img_list.append([0,0,0])
                        
            elif args.slices_select == 'center':
                cent = int(resized_img.shape[0]/2) + np.random.randint(-1,2)
                ver_img_list.append(resized_img[cent-1:cent+2])
                
        elif args.dimension == '3D':
            ver_img_list.append(resized_img)
        
        return ver_img_list



    def crop_roi(self, img, bbox):
        img_roi = img[bbox[0][0] : bbox[1][0],
                    bbox[0][1] : bbox[1][1],
                    bbox[0][2] : bbox[1][2],
                    ].copy()
        return img_roi

    def get_roi_bbox(self, patch_mask, patch_size): 
        fg_idcs = np.where(patch_mask)

        x_min = fg_idcs[0].min()
        x_max = fg_idcs[0].max()
        y_min = fg_idcs[1].min()
        y_max = fg_idcs[1].max()
        z_min = fg_idcs[2].min()
        z_max = fg_idcs[2].max()

        center = [(x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2]        

        x_start = int(center[0] - patch_size[0]/2)
        x_fin = int(center[0] + patch_size[0]/2)
        y_start = int(center[1] - patch_size[1]/2)
        y_fin = int(center[1] + patch_size[1]/2)
        z_start = int(center[2] - patch_size[2]/2)
        z_fin = int(center[2] + patch_size[2]/2)

        # x
        if center[0] < patch_size[0]/2:
            x_start = 0
            x_fin = patch_size[0]
        elif x_fin > patch_mask.shape[0]:
            x_start = patch_mask.shape[0] - patch_size[0]
            x_fin = patch_mask.shape[0]

        # y
        if center[1] < patch_size[1]/2:
            y_start = 0
            y_fin = patch_size[1]
        elif y_fin > patch_mask.shape[1]:
            y_start = patch_mask.shape[1] - patch_size[1]
            y_fin = patch_mask.shape[1]

        # z
        if center[2] < patch_size[2]/2:
            z_start = 0
            z_fin = patch_size[2]
        elif z_fin > patch_mask.shape[2]:
            z_start = patch_mask.shape[2] - patch_size[2]
            z_fin = patch_mask.shape[2]

        return (x_start, y_start, z_start), (x_fin, y_fin, z_fin)
    
    
class CLSDataset(MultitaskDataset):
    def __init__(self, args, transform=None, subset='train'):
        super().__init__(args, transform, subset)
        self.subset = subset
        self.transform = transform
        self.args = args
        
        if self.args.cls_enabled:
            self.data_path = args.cls_data
        else:
            self.data_path = args.pred_data
    
        split = args.split        
        data = pd.read_csv(self.data_path)
        data = data[data[f'train_{split}'] == (subset == 'train')]
        
        self.dataset = self.create_dataset(data)

    def __getitem__(self, index):
        sample = {}
        if self.args.cls_enabled:
            sample = self.dataset[index]
            if self.args.sort_slice:
                v_idx = sorted(random.sample(range(len(sample['ver_path_list'])), 3))
                v_list = np.array(sample['ver_path_list'])[v_idx]
            else:
                v_list = random.sample(sample['ver_path_list'], 3)
                
            del sample['ver_path_list']
            
            ver_img_list = []
            for ver_path in v_list:
                ver_img_list = self.load_ver_img_list(ver_img_list, ver_path, self.args, self.transform, channels_num_2d=self.args.channels_num_2d)

            vers_img_arr = np.array(ver_img_list)
            vers_img_tensor = torch.from_numpy(vers_img_arr)
            sample['vers_img_tensor'] = vers_img_tensor
            
        return sample

    
    def get_sample_dict(self, idx, data, max_fracture_num):
        sample = {}
        
        raw_path = data.loc[idx, 'path']
    
        ver_list = ['vertebrae_T12.hdf5', 
                    'vertebrae_L1.hdf5', 
                    'vertebrae_L2.hdf5', 
                    'vertebrae_L3.hdf5', 
                    'vertebrae_L4.hdf5']
        
        if self.data_path == self.args.cls_data:
            except_ver_idx = [False, False, False, False, False]
            
            pid = raw_path.split('/')[-1].split('_')[-2]
            pat_data = data[data['case number'].astype(int) == int(pid)]
            try:
                except_ver_idx = pat_data.values[0][1:6] >= 1
            except:
                print(raw_path)
                print(pid)
                print(pat_data)
                return sample
            except_ver_list = list(np.array(ver_list)[except_ver_idx])            
            ver_list = [ver for ver in ver_list if ver not in except_ver_list]
            
            if len(except_ver_list) >= max_fracture_num:
                return sample
            
        ver_path_list = []
        for ver in ver_list:
            ver_path = raw_path.replace('nifti', 'MIL_input_2').split('.')[0] + f'/{ver}'
                
            if not os.path.isfile(ver_path):
                return sample

            # with h5py.File(ver_path, 'r') as f:
            #     ver_mask = np.array(f['vertebrae_mask'])

            #     if not ver_mask.sum():
            #         print(f"{ver_path} has no mask")  
            #         return sample         

            ver_path_list.append(ver_path)
            
        
        if len(ver_path_list) < 3:
            print(f"{raw_path.split('/')[-1].split('_')[-2]} has {len(ver_path_list)} vertes")
            return sample
        
        sample['ver_path_list'] = ver_path_list
        sample['y'] = data.loc[idx, 'y']
        sample['path'] = data.loc[idx, 'path']
        
        return sample
        

class PREDDataset(MultitaskDataset):
    def __init__(self, args, transform=None, subset='train'):
        super().__init__(args, transform, subset)
        self.subset = subset
        self.transform = transform
        self.args = args
                
        if self.args.pred_enabled:
            self.data_path = args.pred_data
        else:
            self.data_path = args.cls_data
    
        split = args.split        
        data = pd.read_csv(self.data_path)
        data = data[data[f'train_{split}'] == (subset == 'train')]
        
        self.dataset = self.create_dataset(data)

    def __getitem__(self, index):
        sample = {}
        if self.args.pred_enabled:
            sample = self.dataset[index]
            v_list = random.sample(sample['ver_path_list'], 3)
            del sample['ver_path_list']
            
            ver_img_list = []
            for ver_path in v_list:
                ver_img_list = self.load_ver_img_list(ver_img_list, ver_path, self.args, self.transform, channels_num_2d=self.args.channels_num_2d)

            vers_img_arr = np.array(ver_img_list)
            vers_img_tensor = torch.from_numpy(vers_img_arr)
            sample['vers_img_tensor'] = vers_img_tensor
            
        return sample
    
    def get_sample_dict(self, idx, data, max_fracture_num):
        sample = {}
        
        raw_path = data.loc[idx, 'path']
    
        ver_list = ['vertebrae_T12.hdf5', 
                    'vertebrae_L1.hdf5', 
                    'vertebrae_L2.hdf5', 
                    'vertebrae_L3.hdf5', 
                    'vertebrae_L4.hdf5']
        
        if self.data_path == self.args.cls_data:
            except_ver_idx = [False, False, False, False, False]
            
            pid = raw_path.split('/')[-1].split('_')[-2]
            pat_data = data[data['case number'] == int(pid)]
            except_ver_idx = pat_data.values[0][1:6] >= 1
            
            except_ver_list = list(np.array(ver_list)[except_ver_idx])            
            ver_list = [ver for ver in ver_list if ver not in except_ver_list]
            
            if len(except_ver_list) >= max_fracture_num:
                return sample
            
        ver_path_list = []
        for ver in ver_list:
            ver_path = raw_path.replace('nifti', 'MIL_input_2').split('.')[0] + f'/{ver}'
                
            if not os.path.isfile(ver_path):
                # return sample
                continue

            # with h5py.File(ver_path, 'r') as f:
            #     ver_mask = np.array(f['vertebrae_mask'])

            #     if not ver_mask.sum():
            #         print(f"{ver_path} has no mask")  
            #         return sample         

            ver_path_list.append(ver_path)
            
        
        if len(ver_path_list) < 3:
            print(f"{raw_path.split('/')[-1].split('_')[-2]} has {len(ver_path_list)} vertes")
            return sample
        
        sample['ver_path_list'] = ver_path_list
        sample['y'] = data.loc[idx, 'y']
        sample['path'] = data.loc[idx, 'path']
        
        if self.args.pred_enabled:
            
            duration = data.loc[idx, 'duration'] # years_to_cancer or years_to_last_followup
            y_seq = np.zeros(self.args.max_followup)
            
            if sample['y']: # someday occur fracture
                years_to_cancer = duration
                sample['y'] = (years_to_cancer < self.args.max_followup)
                
                if sample['y']: # observe fracture within max_followup 
                    time_at_event = years_to_cancer
                    y_seq[years_to_cancer:] = 1
                    y_mask = np.array([1] * (years_to_cancer + 1) + \
                        [0] * (self.args.max_followup - (years_to_cancer + 1)))
                    
                else:
                    years_to_last_followup = duration 
                    
            else:
                years_to_last_followup = duration
                    
            if not sample['y']: # not observe fracture within max_followup 
                time_at_event = min(years_to_last_followup, self.args.max_followup - 1)
                y_mask = np.array(
                    [1] * (time_at_event + 1)
                    + [0] * (self.args.max_followup - (time_at_event + 1)))
                
            if time_at_event > 9:
                print(time_at_event, sample['y'], duration)
            
            sample['y_seq'] = y_seq
            sample['y_mask'] = y_mask
            sample['time_at_event'] = time_at_event
        
        return sample