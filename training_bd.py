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
            poison_percentage, hdlyr_size: int, trigger_size:
        tuple[int, int] = (4, 4)) -> None:
    print(
        f'This is the run for experiment number {exp_num} of {elm_type} on {dataset} dataset with {trigger_type} '
        f'and hidden layer size {hdlyr_size} and trigger size {trigger_size} and target label {target_label} '
        f'and poison percentage {poison_percentage}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_accuracy, bd_test_accuracy = -1, -1  # default values
    elapsed_time = -1

    csv_path = saving_path.joinpath(f'results_bd_{dataset}_{elm_type}.csv')
    if not csv_path.exists():
        csv_path.touch()
        with open(file=csv_path, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['EXPERIMENT_NUMBER', 'ELM_TYPE',
                                 'DATASET', 'HIDDEN_LYR_SIZE', 'TRIGGER_TYPE', 'TARGET_LABEL', 'POISON_PERCENTAGE',
                                 'TRIGGER_SIZE', 'TEST_ACCURACY', 'BD_TEST_ACCURACY',
                                 'TIME_ELAPSED'])
    obj_to_save = None
    obj_path = saving_path.joinpath('saved_models')
    if not obj_path.exists():
        obj_path.mkdir(parents=True)
    obj_path = obj_path.joinpath(
        f'{exp_num}_{dataset}_{elm_type}_{trigger_type}_{target_label}_{poison_percentage}_{hdlyr_size}_{trigger_size[0]}.pkl')

    ds_dict = {'mnist': mnist, 'fmnist': fmnist, 'cifar10': cifar10, 'svhn': svhn}

    all_data = ds_dict[dataset].get_alldata_backdoor(target_label=target_label,
                                                     train_samples_percentage=poison_percentage,
                                                     trigger_size=trigger_size)

    if elm_type.lower() == 'poelm':
        poelm = elm.ELMClassifier(hidden_layer_size=hdlyr_size)
        start_time = time.time()
        poelm.fit(all_data['bd_train']['x'], all_data['bd_train']['y_oh'])
        elapsed_time = time.time() - start_time
        out = poelm.predict(all_data['test']['x'])
        test_accuracy = torch.sum(all_data['test']['y'] == torch.from_numpy(out)).item() / len(out)
        bd_out = poelm.predict(all_data['bd_test']['x'])
        bd_test_accuracy = torch.sum(all_data['bd_test']['y'] == torch.from_numpy(bd_out)).item() / len(bd_out)
        obj_to_save = poelm
        del poelm, out, bd_out

    elif elm_type.lower() == 'elm-pca':
        pct = pca_transformed.PCTClassifier(hidden_layer_size=hdlyr_size,
                                            retained=None)  # retained can be (0, 1) percent variation or an integer number of PCA modes to retain
        start_time = time.time()
        pct.fit(all_data['bd_train']['x'], all_data['bd_train']['y_oh'])
        elapsed_time = time.time() - start_time
        out = pct.predict(all_data['test']['x'])
        test_accuracy = torch.sum(all_data['test']['y'] == torch.from_numpy(out)).item() / len(out)
        bd_out = pct.predict(all_data['bd_test']['x'])
        bd_test_accuracy = torch.sum(all_data['bd_test']['y'] == torch.from_numpy(bd_out)).item() / len(bd_out)
        del pct, out, bd_out

    elif elm_type.lower() == 'pca-elm':
        pci = pca_initialization.PCIClassifier(
            retained=None)  # retained can be (0, 1) percent variation or an integer number of PCA modes to retain
        start_time = time.time()
        pci.fit(all_data['bd_train']['x'], all_data['bd_train']['y_oh'])
        elapsed_time = time.time() - start_time
        out = pci.predict(all_data['test']['x'])
        test_accuracy = torch.sum(all_data['test']['y'] == torch.from_numpy(out)).item() / len(out)
        bd_out = pci.predict(all_data['bd_test']['x'])
        bd_test_accuracy = torch.sum(all_data['bd_test']['y'] == torch.from_numpy(bd_out)).item() / len(bd_out)
        del pci, out, bd_out

    elif elm_type.lower() == 'pruned-elm':
        prune = pruned_elm.PrunedClassifier(hidden_layer_size=hdlyr_size)
        start_time = time.time()
        prune.fit(all_data['bd_train']['x'], all_data['bd_train']['y_oh'])
        elapsed_time = time.time() - start_time
        out = prune.predict(all_data['test']['x'])
        test_accuracy = torch.sum(all_data['test']['y'] == torch.from_numpy(out)).item() / len(out)
        bd_out = prune.predict(all_data['bd_test']['x'])
        bd_test_accuracy = torch.sum(all_data['bd_test']['y'] == torch.from_numpy(bd_out)).item() / len(bd_out)
        del prune, out, bd_out

    elif elm_type.lower() == 'drop-elm':
        drop = drop_elm.DropClassifier(hidden_layer_size=hdlyr_size, dropconnect_pr=0.3, dropout_pr=0.3,
                                       dropconnect_bias_pctl=None, dropout_bias_pctl=None)
        start_time = time.time()
        drop.fit(all_data['bd_train']['x'], all_data['bd_train']['y_oh'])
        elapsed_time = time.time() - start_time
        out = drop.predict(all_data['test']['x'])
        test_accuracy = torch.sum(all_data['test']['y'] == torch.from_numpy(out)).item() / len(out)
        bd_out = drop.predict(all_data['bd_test']['x'])
        bd_test_accuracy = torch.sum(all_data['bd_test']['y'] == torch.from_numpy(bd_out)).item() / len(bd_out)
        obj_to_save = drop
        del drop, out, bd_out

    elif elm_type.lower() == 'drelm':
        acc_train, test_accuracy, final_standard_div, (
            W_list, Beta_list, W_prime_list), elapsed_time = DRELM_main.DRELM_main(all_data['bd_train']['x'],
                                                                                   all_data['bd_train']['y_oh'].numpy(),
                                                                                   all_data['test']['x'],
                                                                                   all_data['test']['y_oh'].numpy(),
                                                                                   hidden_size=hdlyr_size)

        bd_test_accuracy = DRELM_main.DRELM_test(X_test=all_data['bd_test']['x'],
                                                 Y_test=all_data['bd_test']['y_oh'].numpy(),
                                                 n_hid=hdlyr_size, W_list=W_list, Beta_list=Beta_list,
                                                 W_prime_list=W_prime_list)
        del W_list, Beta_list, W_prime_list



    elif elm_type.lower() == 'telm':
        test_accuracy, acc_train, (Wie, Whe, Beta_new), elapsed_time, param = TELM_Main.TELM_main(
            all_data['bd_train']['x'],
            all_data['bd_train'][
                'y_oh'].numpy(),
            all_data['test']['x'],
            all_data['test'][
                'y_oh'].numpy(),
            hidden_size=hdlyr_size)
        bd_test_accuracy = TELM_Main.TELM_test(X_test=all_data['bd_test']['x'],
                                               Y_test=all_data['bd_test']['y_oh'].numpy(),
                                               Wie=Wie, Whe=Whe, Beta_new=Beta_new)
        obj_to_save = param
        del Wie, Whe, Beta_new, param

    elif elm_type.lower() == 'mlelm':
        test_accuracy, (betahat_1, betahat_2, betahat_3, betahat_4), elapsed_time, params = ML_ELM_main.main_ML_ELM(
            all_data['bd_train']['x'],
            all_data['bd_train']['y_oh'].numpy(),
            all_data['test']['x'],
            all_data['test']['y_oh'].numpy(),
            hidden_layer=hdlyr_size)
        bd_test_accuracy = ML_ELM_main.MLELM_test(X_test=all_data['bd_test']['x'],
                                                  Y_test=all_data['bd_test']['y_oh'].numpy(),
                                                  betahat_1=betahat_1, betahat_2=betahat_2, betahat_3=betahat_3,
                                                  betahat_4=betahat_4)
        obj_to_save = params
        del betahat_1, betahat_2, betahat_3, betahat_4, params

    elif elm_type.lower() == 'cnn-elm':
        dataloaders, classes_names = ds_dict[dataset].get_dataloaders_backdoor(batch_size=30000, drop_last=False,
                                                                               is_shuffle=True,
                                                                               target_label=target_label,
                                                                               train_samples_percentage=poison_percentage,
                                                                               trigger_size=trigger_size)

        model = main_CNNELM.Net()
        model.to(device)
        optimizer = pseudoInverse.pseudoInverse(params=model.parameters(), C=1e-3)
        start_time = time.time()
        main_CNNELM.train(model, optimizer, dataloaders['bd_train'])
        elapsed_time = time.time() - start_time
        main_CNNELM.train_accuracy(model, dataloaders['bd_train'])
        test_accuracy = main_CNNELM.test(model, dataloaders['test']).item()
        bd_test_accuracy = main_CNNELM.test(model, dataloaders['bd_test']).item()

    with open(file=csv_path, mode='a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [exp_num, elm_type, dataset, hdlyr_size, trigger_type, target_label, poison_percentage, trigger_size,
             test_accuracy, bd_test_accuracy, elapsed_time])

    if obj_to_save is None:
        raise ValueError('obj_to_save is None')
    with open(obj_path, 'wb') as file:
        pickle.dump(obj_to_save, file)

    del all_data
    gc.collect()
