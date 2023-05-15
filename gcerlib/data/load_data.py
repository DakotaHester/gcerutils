import torchgeo
import os
from pathlib import Path

from gcerlib.data import tiff_loader, img_loader, sampling
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png"]

def get_dataset(config):
    
    # check if dataset is tif/tiff
    
    data_srcs = determine_dataset_directory_structure(config)
    
    extension = os.path.splitext(config.dataset.inputs.filename_glob)[1]
    if extension is "tiff":
        pass
        dataset = tiff_loader(config)
    
    if extension in SUPPORTED_IMAGE_FORMATS:
        dataset = img_loader(config) 


def determine_dataset_directory_structure(config):
    root = Path(config.dataset.dir)
    subdirs = [str(dir) for dir in root.glob('*/') if dir.is_dir()]
    
    print(subdirs)
    
    # case 1: root contains all images (no subfolders for split and/or input/target
    # here, we use the filename glob to determine input/target files
    if not subdirs:
        print('CASE 1\n'+ str(root))
        return None # handle this case in the loader
    
    subsubdirs = subdirs = [str(dir) for dir in Path(subdirs[0]).glob('*/') if dir.is_dir()]
    
    if not subsubdirs:
        # case 2: root contains split XOR target/input subfolders
        if config.dataset.splits:
            # case 2a: folders split by test/train/val, but no subfolders for input/target
            print('CASE 2a\n'+ str(root))
            dirs = {split: str(root / split) for split in config.dataset.splits}
            print(dirs)
            return dirs
        else:
            # case 2b: folders for input/target, but no split subfolders
            print('CASE 2b\n'+ str(root))
            dirs = {io: str(root / io) for io in config.dataset.gt_target_folders}
            print(dirs)
            return dirs
    # case 3: root contains subfolders for split and input/target
    # need to determine order
    if set(subdirs) & set(config.dataset.splits):
        # case 3a: split folders and input/target subfolders
        print('CASE 3a\n'+ str(root))
        dirs = {split: {io: str(root / split / io) for io in config.dataset.gt_target_folders} for split in config.dataset.splits}
        print(dirs)
        return dirs
    else:
        # case 3b: input/target folders and split subfolders
        print('CASE 3b\n'+ str(root))
        dirs = {split: {io: str(root / io / split) for io in config.dataset.gt_target_folders} for split in config.dataset.splits}
        print(dirs)
        return dirs
