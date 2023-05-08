import torchgeo
import os

SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png"]

def get_dataset(config):
    
    # check if dataset is tif/tiff
    extension = os.path.splitext(config.dataset.inputs.filename_glob)[1]
    print(extension)