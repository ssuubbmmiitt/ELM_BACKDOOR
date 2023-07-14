import pathlib

from elm_versions import elm, pca_transformed, pca_initialization, pruned_elm, drop_elm
from elm_versions import DRELM_main, TELM_Main, ML_ELM_main
from elm_versions import main_CNNELM, pseudoInverse
from dataset_handler import mnist, fmnist, cifar10, svhn
import csv
import pathlib
import torch
import time
import gc
import pickle


def trainer(exp_num: int, saving_path: pathlib.Path, elm_type: str, dataset: str, trigger_type: str, target_label: int,
            prune_rate: float,
            poison_percentage, hdlyr_size: int, trigger_size:
        tuple[int, int] = (4, 4)) -> None:
    print(
        f'This is the run for experiment number {exp_num} for pruning. Pruning rate is {prune_rate}. Experiment is of {elm_type} on {dataset} dataset with {trigger_type} '
        f'and hidden layer size {hdlyr_size} and trigger size {trigger_size} and target label {target_label} '
        f'and poison percentage {poison_percentage}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_accuracy, bd_test_accuracy = -1, -1  # default values
    elapsed_time = -1

    csv_path = saving_path.joinpath(f'results_pruning_{dataset}_{elm_type}.csv')
    if not csv_path.exists():
        csv_path.touch()
        with open(file=csv_path, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['EXPERIMENT_NUMBER', 'ELM_TYPE',
                                 'DATASET', 'HIDDEN_LYR_SIZE', 'PRUNE_RATE', 'TRIGGER_TYPE', 'TARGET_LABEL',
                                 'POISON_PERCENTAGE',
                                 'TRIGGER_SIZE', 'TEST_ACCURACY', 'BD_TEST_ACCURACY',
                                 'TIME_ELAPSED'])


    obj_to_load = None
    obj_path = saving_path.joinpath('saved_models')
    if not obj_path.exists():
        raise FileNotFoundError(f'No directory called {obj_path} exists.')
    obj_path = obj_path.joinpath(
        f'{exp_num}_{dataset}_{elm_type}_{trigger_type}_{target_label}_{poison_percentage}_{hdlyr_size}_{trigger_size[0]}.pkl')
    if not obj_path.exists():
        raise FileNotFoundError(f'No file called {obj_path} exists.')

    with open(obj_path, 'rb') as file:
        obj_to_load = pickle.load(file)
    if obj_to_load is None:
        raise ValueError(f'No object was loaded from {obj_path}.')


    ds_dict = {'mnist': mnist, 'fmnist': fmnist, 'cifar10': cifar10, 'svhn': svhn}

    all_data_clean = ds_dict[dataset].get_alldata_simple()

    all_data_bd = ds_dict[dataset].get_alldata_backdoor(target_label=target_label,
                                                     train_samples_percentage=poison_percentage,
                                                     trigger_size=trigger_size)

    if elm_type.lower() == 'poelm':
        poelm = obj_to_load
        start_time = time.time()
        poelm.fit_with_mask(all_data_clean['train']['x'], all_data_clean['train']['y_oh'], prune_rate=prune_rate)
        elapsed_time = time.time() - start_time
        out = poelm.predict_with_mask(all_data_bd['test']['x'])
        test_accuracy = torch.sum(all_data_bd['test']['y'] == torch.from_numpy(out)).item() / len(out)
        bd_out = poelm.predict_with_mask(all_data_bd['bd_test']['x'])
        bd_test_accuracy = torch.sum(all_data_bd['bd_test']['y'] == torch.from_numpy(bd_out)).item() / len(bd_out)
        del poelm, out, bd_out


    elif elm_type.lower() == 'drop-elm':
        drop = obj_to_load
        start_time = time.time()
        drop.fit_with_mask(all_data_clean['train']['x'], all_data_clean['train']['y_oh'], prune_rate=prune_rate)
        elapsed_time = time.time() - start_time
        out = drop.predict_with_mask(all_data_bd['test']['x'])
        test_accuracy = torch.sum(all_data_bd['test']['y'] == torch.from_numpy(out)).item() / len(out)
        bd_out = drop.predict_with_mask(all_data_bd['bd_test']['x'])
        bd_test_accuracy = torch.sum(all_data_bd['bd_test']['y'] == torch.from_numpy(bd_out)).item() / len(bd_out)
        del drop, out, bd_out

    elif elm_type.lower() == 'telm':
        param = obj_to_load
        test_accuracy, acc_train, (Wie, Whe, Beta_new), elapsed_time, prune_mask = TELM_Main.TELM_main_with_mask(
            all_data_clean['train']['x'],
            all_data_clean['train'][
                'y_oh'].numpy(),
            all_data_bd['test']['x'],
            all_data_bd['test'][
                'y_oh'].numpy(),
            hidden_size=hdlyr_size,
            prune_rate=prune_rate,
        param=param)
        bd_test_accuracy = TELM_Main.TELM_test_with_mask(X_test=all_data_bd['bd_test']['x'],
                                                         Y_test=all_data_bd['bd_test']['y_oh'].numpy(),
                                                         Wie=Wie, Whe=Whe, Beta_new=Beta_new, prune_mask=prune_mask)
        del Wie, Whe, Beta_new, prune_mask

    elif elm_type.lower() == 'mlelm':
        params = obj_to_load
        test_accuracy, (
            betahat_1, betahat_2, betahat_3, betahat_4), elapsed_time, prune_mask = ML_ELM_main.main_ML_ELM_with_mask(
            all_data_clean['train']['x'],
            all_data_clean['train']['y_oh'].numpy(),
            all_data_bd['test']['x'],
            all_data_bd['test']['y_oh'].numpy(),
            prune_rate=prune_rate,
            params=params,
            hidden_layer=hdlyr_size)
        bd_test_accuracy = ML_ELM_main.MLELM_test_with_mask(X_test=all_data_bd['bd_test']['x'],
                                                            Y_test=all_data_bd['bd_test']['y_oh'].numpy(),
                                                            betahat_1=betahat_1, betahat_2=betahat_2,
                                                            betahat_3=betahat_3,
                                                            betahat_4=betahat_4, prune_mask=prune_mask)
        del betahat_1, betahat_2, betahat_3, betahat_4

    with open(file=csv_path, mode='a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [exp_num, elm_type, dataset, hdlyr_size, prune_rate, trigger_type, target_label, poison_percentage,
             trigger_size,
             test_accuracy, bd_test_accuracy, elapsed_time])

    del all_data_bd
    gc.collect()
