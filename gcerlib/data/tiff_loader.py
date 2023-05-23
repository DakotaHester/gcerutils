from torchgeo.datasets import RasterDataset
import os

def load_input(config):
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
    
    
