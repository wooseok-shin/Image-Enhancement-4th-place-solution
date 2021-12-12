import os
import pprint
import random
import warnings
import torch
import numpy as np
from config import getConfig

warnings.filterwarnings('ignore')
args = getConfig()

def main(args):
    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    save_path = os.path.join(args.model_path, (args.exp_num).zfill(3))

    # Create model directory
    os.makedirs(save_path, exist_ok=True)

    if args.img_size == 256:
        from trainer_256 import Trainer
        Trainer(args, save_path)
    elif args.img_size == 352:
        from trainer_352 import Trainer        
        Trainer(args, save_path)

if __name__ == '__main__':
    main(args)
