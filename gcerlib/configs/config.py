import sys
from yconf import BaseConfiguration
import os
import datetime
from torch.cuda import is_available
import random
from torch import manual_seed
import numpy as np

def create_output_folder(config):
    # print(args, config)
    out_path = config.experiment_name 
    if config.include_date_in_results_folder:
        out_path = out_path + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if os.path.exists(out_path): print(f'[WARN] {out_path} already exists, results will be overwritten')
    os.makedirs(out_path, exist_ok=True)
    config['out_path'] = out_path
    
    return config

def get_config():
    # quick hack to set default config location
    DEFAULT_CONFIG_FILE = 'gcerlib/configs/base_config.yaml' # os.path.join('gcerlib', 'configs', 'config.yaml')
    args = sys.argv[1:]
    if '-c' not in args and '--config' not in args:
        args.append('-c')
        args.append(DEFAULT_CONFIG_FILE)
    
    if not os.path.isfile(args[args.index('-c') + 1]):
        raise FileNotFoundError(f'Config file {args[args.index("-c") + 1]} does not exist')

    config = BaseConfiguration()
    config.parse(args)
    
    # add device
    config['device'] = 'cuda' if is_available() else 'cpu'
        
    print("[CONFIG] Loaded config from ", config.configPath)

    config = create_output_folder(config)
    set_seeds(config.seed)
    
    return config

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)