import training
from pathlib import Path
import gc
import argparse


parser = argparse.ArgumentParser(description='BASL_Autoencoder')
parser.add_argument('--dataname', type=str, default='fmnist',
                    choices=['mnist', 'svhn', 'fmnist', 'cifar10'],
                    help='The dataset to use')
parser.add_argument('--elmtype', type=str, default='poelm',
                    choices=['poelm', 'drop-elm', 'telm', 'mlelm'],
                    help='elm type to use')
args = parser.parse_args()




# saving_path = Path()
# n_of_experiments = 2
# elm_type_list = ['poelm', 'elm-pca', 'pca-elm', 'pruned-elm', 'drop-elm', 'drelm', 'telm', 'mlelm']
# dataset_list = ['mnist']
# hdlyr_size_list = [500, 700, 1000, 1500, 2000]
#
#
# for dataset in dataset_list:
#     for elm_type in elm_type_list:
#         for hdlyr_size in hdlyr_size_list:
#             for exp_num in range(n_of_experiments):
#                 training.trainer(exp_num=exp_num, saving_path=saving_path, elm_type=elm_type, dataset=dataset, hdlyr_size=hdlyr_size)
#                 gc.collect()


saving_path = Path()
n_of_experiments = 1
# elm_type_list = ['poelm', 'elm-pca', 'pca-elm', 'drop-elm', 'drelm', 'telm', 'mlelm']
# dataset_list = ['mnist']
elm_type_list = [args.elmtype]
dataset_list = [args.dataname]
hdlyr_size_list = [500, 1000, 2000, 5000, 8000]


for dataset in dataset_list:
    for elm_type in elm_type_list:
        for hdlyr_size in hdlyr_size_list:
            for exp_num in range(n_of_experiments):
                training.trainer(exp_num=exp_num, saving_path=saving_path, elm_type=elm_type, dataset=dataset, hdlyr_size=hdlyr_size)
                gc.collect()



