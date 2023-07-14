import torch.utils.data
from torchvision import datasets, transforms
# from torchvision.transforms.functional import rgb_to_grayscale
from typing import Optional, Callable, Any, Tuple

from dataset_handler.trigger import get_backdoor_test_dataset, get_backdoor_train_dataset, GenerateTrigger, \
    toonehottensor

import torch
from PIL import Image


torch.manual_seed(47)
import numpy as np

np.random.seed(47)


# def rgb_to_grayscale(img):
#     r, g, b = img[0], img[1], img[2]
#     gray_img = 0.2990 * r + 0.5870 * g + 0.1140 * b
#     gray_img = gray_img / np.amax(gray_img)
#     # gray_img = gray_img[np.newaxis, ...]
#     # info = np.finfo(gray_img.dtype)  # Get the information of the incoming image type
#     # gray_img = gray_img.astype(np.float64) / info.max  # normalize the data to 0 - 1
#     # gray_img = 255 * gray_img  # Now scale by 255
#     gray_img = (gray_img * 255).astype(np.uint8)
#     return gray_img


class SVHN(datasets.svhn.SVHN):
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target




def get_dataloaders_simple(batch_size, drop_last, is_shuffle):
    drop_last = drop_last
    is_shuffle = is_shuffle
    batch_size = batch_size

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load data
    train_dataset = datasets.svhn.SVHN(root='./data/SVHN', split='train', download=True, transform=transform)
    test_dataset = datasets.svhn.SVHN(root='./data/SVHN', split='test', download=True, transform=transform)


###############################this section is for converting rgb to grayscale and normalizing it#######################


    # Convert RGB images to grayscale

    # new_train_data = [np.squeeze(rgb_to_grayscale(torch.from_numpy(img)).numpy()) for img in train_dataset.data]
    # new_train_data = [rgb_to_grayscale(img) for img in train_dataset.data]
    # new_train_data = np.array(new_train_data, dtype=np.uint8)
    # train_dataset.data = new_train_data

    # new_test_data = [np.squeeze(rgb_to_grayscale(torch.from_numpy(img)).numpy()) for img in test_dataset.data]
    # new_test_data = [rgb_to_grayscale(img) for img in test_dataset.data]
    # new_test_data = np.array(new_test_data, dtype=np.uint8)
    # test_dataset.data = new_test_data


    # # Normalize the grayscale images
    # mean = train_dataset.data.mean()
    # std = train_dataset.data.std()
    # transform = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((mean,), (std,))
    #
    # ])
    #
    # train_dataset.transform = transform
    # test_dataset.transform = transform

    ##############################################################################################################

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=len(train_dataset) if batch_size is None else batch_size,
                                                   shuffle=is_shuffle, num_workers=num_workers,
                                                   drop_last=drop_last)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=len(test_dataset) if batch_size is None else batch_size,
                                                  shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)

    # validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=len(
    #     validation_dataset) if batch_size is None else batch_size,
    #                                                     shuffle=is_shuffle, num_workers=num_workers,
    #                                                     drop_last=drop_last)

    return {'train': train_dataloader,
            'test': test_dataloader}


def get_dataloaders_backdoor(batch_size, drop_last, is_shuffle, target_label, train_samples_percentage,
                             trigger_size=(8, 8)):
    drop_last = drop_last
    batch_size = batch_size
    is_shuffle = is_shuffle
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load data
    train_dataset = datasets.SVHN(root='./data/SVHN', split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root='./data/SVHN', split='test', download=True, transform=transform)

    trigger_obj = GenerateTrigger(trigger_size, pos_label='upper-left', dataset='svhn', shape='square')

    bd_train_dataset = get_backdoor_train_dataset(train_dataset, trigger_obj, trig_ds='svhn',
                                                  samples_percentage=train_samples_percentage,
                                                  backdoor_label=target_label)

    backdoor_test_dataset = get_backdoor_test_dataset(test_dataset, trigger_obj, trig_ds='svhn',
                                                      backdoor_label=target_label)
    # train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset) if batch_size is None else batch_size,
    #                                                shuffle=is_shuffle, num_workers=num_workers,
    #                                                drop_last=drop_last)
    bd_train_dataloader = torch.utils.data.DataLoader(dataset=bd_train_dataset, batch_size=len(
        bd_train_dataset) if batch_size is None else batch_size,
                                                      shuffle=is_shuffle, num_workers=num_workers,
                                                      drop_last=drop_last)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=len(test_dataset) if batch_size is None else batch_size,
                                                  shuffle=is_shuffle, num_workers=num_workers,
                                                  drop_last=drop_last)

    backdoor_test_dataloader = torch.utils.data.DataLoader(dataset=backdoor_test_dataset, batch_size=len(
        backdoor_test_dataset) if batch_size is None else batch_size,
                                                           shuffle=is_shuffle, num_workers=num_workers,
                                                           drop_last=drop_last)

    return {'bd_train': bd_train_dataloader,
            'test': test_dataloader,
            'bd_test': backdoor_test_dataloader}


def get_alldata_simple():
    '''
    a method which calls the dataloaders, and iterate through them,
         flattens the inputs and returns all dataset in just one batch of data.
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloaders = get_dataloaders_simple(batch_size=None, drop_last=True, is_shuffle=False)
    all_data = {item: {} for item in dataloaders.keys()}
    for phase in all_data.keys():
        for i_batch, sample_batched in enumerate(dataloaders[phase]):
            # print(type(sample_batched[0]))
            # print(len(sample_batched[0]))
            # print(sample_batched[0].shape)
            # print(type(sample_batched[1]))
            # print(len(sample_batched[1]))
            # print(sample_batched[1].shape)
            all_data[phase]['x'] = torch.reshape(sample_batched[0], (len(dataloaders[phase].dataset), -1)).to(device)
            all_data[phase]['y'] = sample_batched[1].to(device)
            all_data[phase]['y_oh'] = toonehottensor(10, sample_batched[1]).to(device)

    return all_data


def get_alldata_backdoor(target_label, train_samples_percentage, trigger_size):
    '''
    a method which calls the dataloaders, and iterate through them,
         flattens the inputs and returns all dataset in just one batch of data.
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloaders = get_dataloaders_backdoor(batch_size=None, drop_last=True, is_shuffle=False,
                                           target_label=target_label,
                                           train_samples_percentage=train_samples_percentage,
                                           trigger_size=trigger_size)
    all_data = {item: {} for item in dataloaders.keys()}
    for phase in all_data.keys():
        for i_batch, sample_batched in enumerate(dataloaders[phase]):
            # print(type(sample_batched[0]))
            # print(len(sample_batched[0]))
            # print(sample_batched[0].shape)
            # print(type(sample_batched[1]))
            # print(len(sample_batched[1]))
            # print(sample_batched[1].shape)
            all_data[phase]['x'] = torch.reshape(sample_batched[0], (len(dataloaders[phase].dataset), -1)).to(device)
            all_data[phase]['y'] = sample_batched[1].to(device)
            all_data[phase]['y_oh'] = toonehottensor(10, sample_batched[1]).to(device)

    return all_data
