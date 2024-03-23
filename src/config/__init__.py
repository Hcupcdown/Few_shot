import os

import yaml

from .config import get_config


def get_args_dict():
    """
    Get your arguments and make dictionary.
    If you add some arguments in the model, you should edit here also.
    """

    args = get_config()
    args.dataset_dir = {'train':args.train}
    args.few_shot_dataset = {'trainset_dir':args.train,
                             'few_shot_dir':args.few_shot_dir,
                             'num_shot':args.num_shot,
                             'valset_dir':args.few_shot_val}

    args.ex_name = os.getcwd().replace('\\','/').split('/')[-1]
    model_config_path = os.path.join('src/config/model_config/RadarMossFormer.yml')
    with open(model_config_path) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    args.model_config = model_config

    return args