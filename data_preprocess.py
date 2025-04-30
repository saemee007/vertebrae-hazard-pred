import os
from glob import glob
from tqdm import tqdm

from utils.parsing import parse_argument_data_preprocess
from data.refine_mask import refine_masks
from data.make_mil_input import make_MIL_input_2

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    print()
    print("Start data preprocessing (4 stages) ... \n")

    # Arguments
    args = arg_parse()

    # Run Total Segmentator
    print()
    print(f"=> Load data from {args.data_root}")
    print("Start run of total segmentator (1 / 4 stages) ... ")

    for nii_path in tqdm(glob(os.path.join(args.data_root, f'nifti/*{args.file_type}'))):
        if not os.path.isdir(nii_path.replace('nifti', 'tot_seg_mask')[:-len(args.file_type)]):
            os.system(f"CUDA_VISIBLE_DEVICES={args.gpu_id} TotalSegmentator -i {nii_path} -o {nii_path.replace('nifti', 'tot_seg_mask')[:-len(args.file_type)]}")

    # Run v7 2d U-Net
    if True:
        print()
        print("Start inference of v7_2d_unet (2 / 4 stages) ... ")
        os.system(f"python /home/saemeechoi/snu/v7_2d_unet/inference.py --data_root {args.data_root} --gpu_id {args.gpu_id} --file_type {args.file_type} --config /home/saemeechoi/snu/v7_2d_unet/configs/infer_config.yaml")
    
    # Refine Segmentation
    print()
    print("Start refinement of two segmentation masks (3 / 4 stages) ... ")
    refine_masks(args.data_root)

    # Make MIL_input_2
    print()
    print("Start generation of MIL_input_2 (4 / 4 stages) ... ")
    make_MIL_input_2(args.data_root, args.file_type)    