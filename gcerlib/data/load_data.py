from torchgeo.datasets import RasterDataset
import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import json
import pickle

SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]

def get_dataset(config):
    
    # check if dataset is tif/tiff
    
    
    
    extension = os.path.splitext(config.dataset.inputs.filename_glob)[1]
    print(config.dataset)
    print(extension)
    if extension == ".tiff":
        print('[DATA] Loading from GeoTIFF')
        dataset = tiff_loader(config)
    
    elif extension in SUPPORTED_IMAGE_FORMATS:
        print(f'[DATA] Loading from image ({extension})')
        dataset = img_loader(config) 
    
    print(dataset)


def determine_dataset_directory_structure(config):
    root = Path(config.dataset.dir)
    print(f'[DATA] Loading from {root}')
    subdirs = [str(dir) for dir in root.glob('*/') if dir.is_dir()]
    
    # print(subdirs)
    
    # case 1: root contains all images (no subfolders for split and/or input/target
    # here, we use the filename glob to determine input/target files
    if not subdirs:
        # print('CASE 1\n'+ str(root))
        return root # handle this case in the loader
    
    subsubdirs = subdirs = [str(dir) for dir in Path(subdirs[0]).glob('*/') if dir.is_dir()]
    
    if not subsubdirs:
        # case 2: root contains split XOR target/input subfolders
        if config.dataset.splits:
            # case 2a: folders split by test/train/val, but no subfolders for input/target
            # print('CASE 2a\n'+ str(root))
            dirs = {split: root / split for split in config.dataset.splits}
            # print(dirs)
            return dirs
        else:
            # case 2b: folders for input/target, but no split subfolders
            print('CASE 2b\n'+ str(root))
            dirs = {io: root / io for io in config.dataset.gt_target_folders}
            # print(dirs)
            return dirs
    # case 3: root contains subfolders for split and input/target
    # need to determine order
    if set(subdirs) & set(config.dataset.splits):
        # case 3a: split folders and input/target subfolders
        # print('CASE 3a\n'+ str(root))
        dirs = {split: {io: root / split / io for io in config.dataset.gt_target_folders} for split in config.dataset.splits}
        # print(dirs)
        return dirs
    else:
        # case 3b: input/target folders and split subfolders
        # print('CASE 3b\n'+ str(root))
        dirs = {split: {io: root / io / split for io in config.dataset.gt_target_folders} for split in config.dataset.splits}
        # print(dirs)
        return dirs

def img_loader(config):
    
    # Path object
    directories = determine_dataset_directory_structure(config)
    
    if config.dataset.splits is not None:
        if config.dataset.gt_target_folders is not None:
            # splits and input/target folders
            pass
        else: # splits only
            pass
    else:
        if config.dataset.gt_target_folders is not None:
            folders = config.dataset.gt_target_folders
            # input/target folders only
            input_files = directories[folders[0]].glob(config.dataset.inputs.filename_glob)
            target_files = directories[folders[1]].glob(config.dataset.targets.filename_glob)
            

            X = [np.array(Image.open(input_file)) / 255.0 for input_file in input_files]
            y = [np.array(Image.open(target_file)) for target_file in target_files]
            
            dataset = {}
            
            # split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.dataset.splits_to_use.test[0], random_state=config.seed)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config.dataset.splits_to_use.val[0], random_state=config.seed)
            
            dataset['train'] = {'input': X_train, 'target': y_train}
            dataset['test'] = {'input': X_test, 'target': y_test}
            dataset['val'] = {'input': X_val, 'target': y_val}
            
            print(f'[DATA] Loaded {len(X)} total samples')
            print(f'[DATA] Loaded {len(X_train)} training samples')
            print(f'[DATA] Loaded {len(X_val)} validation samples')
            print(f'[DATA] Loaded {len(X_test)} testing samples')
            
            if config.dataset.save_datasets:
                out_file = os.path.join(config.out_path, 'datasets.pkl')
                print(f'[DATA] Saving datasets to {out_file}')
                pickle.dump(dataset, open(out_file, 'wb'))
            
            return dataset
        # just root directory
        else:
            pass

def tiff_loader(config):
    # TODO
    class TiffDataset(RasterDataset):
        filename_glob=config.dataset.inputs.filename_glob,
        filename_regex=config.dataset.inputs.filename_regex,
        date_format=config.dataset.inputs.date_format,
        is_image=config.dataset.inputs.is_image,
        separate_files=config.dataset.inputs.separate_files,
        all_bands=config.dataset.inputs.all_bands,
        rgb_bands=config.dataset.inputs.rgb_bands,
    
    
    if config.dataset.splits:
        for split in config.dataset.splits:
            if os.path.isdir(os.path.join(config.dataset.inputs.dir, split)):
                print(f'Loading {split} split from {config.dataset.inputs.dir}')
                
    input_dir = config.dataset.inputs.dir
    pass

def split_image_dataset(config, image_dataset):
    # TODO
    pass    
def split_tiff_dataset(config, tiff_dataset):
    # TODO
    pass