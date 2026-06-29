import argparse
import new_types as nt
from collections.abc import Iterable
from configs import Configuration
import os
from pathlib import Path
from trainer import train_with_different_seeds


def main():
    dir_script = os.path.dirname(Path(__file__).resolve())
    current_working_directory = Path.cwd()
    args = obtain_args()
    
    cfg = Configuration(args=args, dir_script=dir_script, current_working_directory=current_working_directory)
    
    train_with_different_seeds(cfg=cfg)


def obtain_args():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorithms.')
    parser.add_argument('config_file', type=str, help='Json file of settings.')
    # Setting the order or gpu_id as an argument in Terminal, overrides the values in the JSON.
    parser.add_argument('-order', type=int, help='Order of domains (it can be set on the JSON file as well)')
    parser.add_argument('-gpu_id', type=int, help='GPU ID')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
