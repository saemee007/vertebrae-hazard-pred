import os
from glob import glob
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk

def refine_mask(unet_mask_arr, target_mask):
    z_idcs = np.unique(np.where(target_mask)[0])
    refined_mask = np.zeros_like(target_mask)

    for z_idx in z_idcs:
        unet_slice = unet_mask_arr[z_idx].copy()
        y_idcs, x_idcs = np.where(unet_slice)

        bool_slice = np.zeros_like(unet_slice)

        for x_idx in x_idcs:
            max_y_idx = np.max(np.where(unet_slice[:, x_idx]))
            bool_slice[:max_y_idx, x_idx] = 1

        refined_mask[z_idx] = np.logical_and(target_mask[z_idx], bool_slice)

    return refined_mask

def refine_masks(data_root):
    for pat in tqdm(glob(os.path.join(data_root, f'tot_seg_mask/*'))):
        patID = os.path.basename(pat)
        os.makedirs(os.path.join(data_root, f'refine_mask_v2/{patID}'),exist_ok=True, mode=0o777)

        unet_mask = sitk.ReadImage(os.path.join(data_root, f'unet_mask/{patID}/body.nii.gz'))
        unet_mask_arr = sitk.GetArrayFromImage(unet_mask)

        for ver_path, ver_name in [(f'{pat}/{ver_name}.nii.gz', ver_name) for ver_name in ['vertebrae_T12', 'vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4']]:
            try:
                if not os.path.isfile(ver_path.replace('tot_seg_mask', 'refine_mask_v2')): 
                    ver = sitk.ReadImage(f'{ver_path}')
                    ver_arr = sitk.GetArrayFromImage(ver)

                    refined_arr = refine_mask(unet_mask_arr[:,::-1], ver_arr[:,::-1,::-1]) 
                    refined_img = sitk.GetImageFromArray(refined_arr)
                    refined_img.CopyInformation(unet_mask)
                    refined_img = sitk.Cast(refined_img, sitk.sitkUInt8)

                    sitk.WriteImage(refined_img, ver_path.replace('tot_seg_mask', 'refine_mask_v2'))

            except:
                print("Error", patID, ver_name)