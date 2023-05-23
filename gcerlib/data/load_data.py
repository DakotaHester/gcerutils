from torchgeo.datasets import RasterDataset
import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import json
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms

SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]

class ImageDictLoader(Dataset):
    def __init__(self, dataset, config, mode='train'):
        super().__init__()
        
        self.dataset = dataset
        self.normalize = config.dataset.normalize
        self.device = config.device
        
        # dataset is in batch, h, w, c format
        # manipulate to batch, c, h, w format
        for key in self.dataset.keys():
            self.dataset[key] = np.moveaxis(self.dataset[key], -1, 1)
        
        self.transform_pipeline = self.transforms()
        
    def transforms(self):
        transforms_list = []
        if self.normalize:
            self.get_stats()
            transforms_list.append(transforms.Normalize(self.mean, self.std))

        return transforms.Compose(transforms_list)
        
    def get_stats(self):

        self.mean = np.mean(self.dataset['input'], axis=(0, 2, 3))
        self.std = np.std(self.dataset['input'], axis=(0, 2, 3))
        
        print(self.mean, self.std)

    def __getitem__(self, index):
        
        # NO PREPROCESSING FOR NOW
        # TODO add preprocessing
        # memory pinning?
        input = torch.tensor(self.dataset['input'][index], dtype=torch.double)
        if self.normalize:
            input = self.transform_pipeline(input)
        target = torch.tensor(self.dataset['target'][index], dtype=torch.uint8)
        
        return input, target
    
    def __len__(self):
        return len(self.dataset['input'])
    
    def inverse_norm(self, images):
        images = images * self.std[np.newaxis, :, np.newaxis, np.newaxis] + self.mean[np.newaxis, :, np.newaxis, np.newaxis]
        return images
    

def get_dataset(config):
    
    if config.dataset.load_saved_dataset:
        try:
            out_file = os.path.join(config.out_path, 'datasets.pkl')
            print(f'[DATA] Loading saved dataset from {out_file}')
            if not os.path.exists(config.out_path):
                raise FileNotFoundError(f'Could not find {out_file}')
            dataset = pickle.load(open(out_file, 'rb'))
        except FileNotFoundError as e:
            print(e)
            print('[DATA] Could not find saved dataset. Loading from source.')
            config.dataset.load_saved_dataset = False
    
    if not config.dataset.load_saved_dataset:
        # check if dataset is tif/tiff
        extension = os.path.splitext(config.dataset.inputs.filename_glob)[1]
        
        if extension == ".tiff":
            print('[DATA] Loading from GeoTIFF')
            dataset = tiff_loader(config)    
        elif extension in SUPPORTED_IMAGE_FORMATS:
            print(f'[DATA] Loading from image ({extension})')
            dataset = img_loader(config) 
        
    if config.dataset.save_datasets:
        out_file = os.path.join(config.out_path, 'datasets.pkl')
        print(f'[DATA] Saving datasets to {out_file}')
        pickle.dump(dataset, open(out_file, 'wb'))
        
    if config.dataset.binary_class and config.model.classes == 1:
        for split in dataset.keys():
            dataset[split]['target'] = np.array([np.where(mask == config.dataset.binary_class, 1, 0) for mask in dataset[split]['target']])

    config['dataset']['image_size'] = dataset['train']['input'][0].shape[1]
    return dataset

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
            # print('CASE 2b\n'+ str(root))
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