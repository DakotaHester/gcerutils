# from torchgeo.datasets import RasterDataset
import os
import rasterio
from pathlib import Path
from rasterio.plot import show
from numpy import np

def get_tiff_dataset(path, input_glob, output_glob):

    root = Path(path)
    input_paths = sorted(root.rglob(input_glob))
    target_paths = sorted(root.rglob(output_glob))
    
    input_tiffs = [rasterio.open(str(p)) for p in input_paths]
    target_tiffs = [rasterio.open(str(p)) for p in target_paths]
    
    print(f'[DATA] Loaded .tiff from {path}')
    
    show(input_tiffs[2].read((1, 2, 3)))
    show(input_tiffs[2].read((4, 3, 1)))
    show(target_tiffs[2].read())
    
    X = np.array([tiff.read() for tiff in input_tiffs])
    y = np.array([tiff.read() for tiff in target_tiffs])
    
    return X, y
        
    

def main():
    get_tiff_dataset(path = os.path.join('test_datasets', 'tiffs_sampled'), input_glob = '*img_*.tif', output_glob = '*label_*.tif')
    
if __name__ == '__main__':
    main()