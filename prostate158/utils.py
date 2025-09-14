import os
import yaml
import torch
import monai
import munch 

def load_config(fn: str='config.yaml'):
    "Load config from YAML and return a serialized dictionary object"
    with open(fn, 'r', encoding='utf-8') as stream:
        config=yaml.safe_load(stream)
    config=munch.munchify(config)
    
    if not config.overwrite:
        i=1
        while os.path.exists(config.run_id+f'_{i}'):
            i+=1
        config.run_id+=f'_{i}'

    config.out_dir = os.path.join(config.run_id, config.out_dir)
    config.log_dir = os.path.join(config.run_id, config.log_dir)
    
    if not isinstance(config.data.image_cols, (tuple, list)): 
        config.data.image_cols=[config.data.image_cols]
    if not isinstance(config.data.label_cols, (tuple, list)): 
        config.data.label_cols=[config.data.label_cols]
    
    config.transforms.mode=('bilinear', ) * len(config.data.image_cols) + \
                             ('nearest', ) * len(config.data.label_cols)
    return config

    
def num_workers():
    "Get max supported workers -2 for multiprocessing"
    return 0

USE_AMP=True if monai.utils.get_torch_version_tuple() >= (1, 6) else False