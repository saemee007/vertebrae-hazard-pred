import os
from glob import glob
from tqdm import tqdm

import h5py
import numpy as np
import SimpleITK as sitk

def get_roi_bbox(patch_mask, patch_size): 

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

def crop_roi(img, bbox):
    # img: D x H x W 
    # bbox: left-top (x1, y1a, z1) and right_bottom (x2, y2, z2)
    
    img_roi = img[bbox[0][0] : bbox[1][0],
                  bbox[0][1] : bbox[1][1],
                  bbox[0][2] : bbox[1][2],
                  ].copy()
    return img_roi

def resample_img(sitk_volume, new_spacing):
    """1) Create resampler"""
    resample = sitk.ResampleImageFilter() 
    
    """2) Set parameters"""
    #set interpolation method, output direction, default pixel value
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(sitk_volume.GetDirection())
    
    #set output spacing
    new_spacing = np.array(new_spacing)
    resample.SetOutputSpacing(new_spacing)
    
    #set output size and origin
    old_size = np.array(sitk_volume.GetSize())
    old_spacing = np.array(sitk_volume.GetSpacing())
    new_size = np.int16(np.ceil(old_size*old_spacing/new_spacing))
    new_origin = np.array(sitk_volume.GetOrigin())
        
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetOutputOrigin(new_origin)
    
    """3) execute"""
    new_volume = resample.Execute(sitk_volume)
    return new_volume

def resample_mask(sitk_volume, new_spacing, new_size=None):
    """1) Create resampler"""
    resample = sitk.ResampleImageFilter() 
    
    """2) Set parameters"""
    #set interpolation method, output direction, default pixel value
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetOutputDirection(sitk_volume.GetDirection())
    
    #set output spacing
    new_spacing = np.array(new_spacing)
    resample.SetOutputSpacing(new_spacing)
    
    #set output size and origin
    old_size = np.array(sitk_volume.GetSize())
    old_spacing = np.array(sitk_volume.GetSpacing())
    new_origin = np.array(sitk_volume.GetOrigin())
    
    if new_size is None:
        new_size = np.int16(np.ceil(old_size*old_spacing/new_spacing))
        new_size = [int(s) for s in new_size]

    resample.SetSize(new_size)
    resample.SetOutputOrigin(new_origin)
    
    """3) execute"""
    new_volume = resample.Execute(sitk_volume)
    return new_volume

def make_MIL_input_2(data_root, file_type):
    patch_size = (24, 128, 224)

    for img_path in tqdm(glob(os.path.join(data_root, f'nifti/*{file_type}'))):
        
        templete = sitk.ReadImage(img_path)
        templete = resample_img(templete, (1., 1., 3.))
        
        img = sitk.GetArrayFromImage(templete)

        ver_list = ['vertebrae_T12', 
                    'vertebrae_L1', 
                    'vertebrae_L2', 
                    'vertebrae_L3', 
                    'vertebrae_L4']

        ver_paths = [f"{img_path.replace('nifti', 'refine_mask_v2').split('.')[0]}/{ver}{file_type}" for ver in ver_list]
        auto_paths = img_path.replace('nifti', 'tot_seg_mask').split('.')[0] + '/autochthon_*'
        ilio_paths = img_path.replace('nifti', 'tot_seg_mask').split('.')[0] + '/iliopsoas_*'

        auto_masks = np.zeros(img.shape)
        for auto_path in glob(auto_paths):
            auto_mask = sitk.GetArrayFromImage(resample_mask(sitk.ReadImage(auto_path), (1., 1., 3.), templete.GetSize()))
            auto_masks[auto_mask.astype(bool)] = 0
            auto_masks += auto_mask

        auto_masks = auto_masks[:,::-1,::-1]
        
        ilio_masks = np.zeros(img.shape)
        for ilio_path in glob(ilio_paths):
            ilio_mask = sitk.GetArrayFromImage(resample_mask(sitk.ReadImage(ilio_path), (1., 1., 3.), templete.GetSize()))
            ilio_masks[ilio_mask.astype(bool)] = 0
            ilio_masks += ilio_mask    

        ilio_masks = ilio_masks[:,::-1,::-1]

        muscle_mask = auto_masks.astype(np.uint8) | ilio_masks.astype(np.uint8)

        for ver_path in ver_paths:
            
            dir_path = f"{img_path.replace('nifti', 'MIL_input_2').split('.')[0]}"
            os.makedirs(dir_path, exist_ok=True, mode=0o777)
            
            ver_name = os.path.basename(ver_path).split('.')[0]
            save_path = f'{dir_path}/{ver_name}.hdf5'

            if os.path.isfile(save_path):
                continue
                
            else:
                patch_mask = np.zeros(img.shape)
                ver_mask = sitk.GetArrayFromImage(resample_mask(sitk.ReadImage(ver_path), (1., 1., 3.), templete.GetSize()))
                ver_mask = ver_mask.astype(np.uint8)
                ver_xyz = np.where(ver_mask == 1)

                if len(ver_xyz[0]):
                    z_min = ver_xyz[0].min()
                    z_max = ver_xyz[0].max()

                    patch_mask[ver_mask.astype(bool)] = 0
                    patch_mask += ver_mask

                    patch_mask[auto_masks.astype(bool)] = 0
                    patch_mask[z_min:z_max,:,:] += auto_masks[z_min:z_max,:,:]

                    patch_mask[ilio_masks.astype(bool)] = 0
                    patch_mask[z_min:z_max,:,:] += ilio_masks[z_min:z_max,:,:]

                else: 
                    print('There is no vertebrae mask')
                    print(ver_path)
                    continue

                if patch_mask.sum():
                    bbox = get_roi_bbox(patch_mask, patch_size)
                    img_roi    = crop_roi(img, bbox)
                    muscle_roi = crop_roi(muscle_mask, bbox)
                    ver_roi    = crop_roi(ver_mask, bbox)

                    if img_roi.shape != patch_size:
                        print('image shape looks weird')
                        print(img_path, img.shape, img_roi.shape, bbox)

                else: # 환자 ct에 해당 mask가 없으면
                    print('There is no patch mask')
                    print(img_path)
                    continue

                # filp 추가되야 함
                
                if True:
                    with h5py.File(save_path, "w") as f:
                        f.create_dataset('image', data=img_roi, dtype=np.float32)
                        f.create_dataset('muscle_mask', data=muscle_roi, dtype=np.uint8)
                        f.create_dataset('vertebrae_mask', data=ver_roi, dtype=np.uint8)