import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100

# Loads the dataset's train set and test set. The resulted train set is the raw train set, without splitting to the validation set.
def get_full_dataset(config):
    dataset_mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4865, 0.4409),
    }
    dataset_std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2673, 0.2564, 0.2762),
    }
    if config.dataset == 'CIFAR-10':
        transform = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Normalize(dataset_mean['cifar10'], dataset_std['cifar10'])
            ])
        full_trainset = CIFAR10("../dataset", train=True, download=True, transform=transform)
        testset = CIFAR10("../dataset", train=False, download=True, transform=transform)

    if config.dataset == 'CIFAR-100':
        transform = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Normalize(dataset_mean['cifar10'], dataset_std['cifar100'])
            ])
        full_trainset = CIFAR100("../dataset", train=True, download=True, transform=transform)
        testset = CIFAR100("../dataset", train=False, download=True, transform=transform)

    return full_trainset, testset

def split_validation_set(config, full_trainset):
    lengths = [len(full_trainset) * (1 - config.dataset.validation_percentage), len(full_trainset) * config.dataset.validation_percentage]
    split_trainset, valset = random_split(full_trainset, lengths) # TODO: maybe add seed option? 
    return split_trainset, valset

def get_full_trainloader(config, full_train_set):
    return DataLoader(full_train_set, batch_size=config.batch_size, shuffle=True) #TODO: config.batch_size

def get_splitted_trainloader(config, splitted_train_set):
    return DataLoader(splitted_train_set, batch_size=config.batch_size, shuffle=True) #TODO: config.batch_size

def get_valloader(config, val_set):
    return DataLoader(val_set, batch_size=config.batch_size, shuffle=True) #TODO: config.batch_size

def get_testloader(config, test_set):
    return DataLoader(test_set, batch_size=config.batch_size, shuffle=True) #TODO: config.batch_size

def get_federated_client_dataloaders(config, full_train_set):
    # Split training set into the number of the clients to simulate the individual dataset
    partition_size = len(full_train_set) // config.num_clients #TODO: check config
    lengths = [partition_size] * config.num_clients #TODO: check config
    datasets= random_split(full_train_set, lengths) # TODO: maybe add seed option? 

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for idx, ds in enumerate(datasets):
        lengths = [len(ds) * (1 - config.dataset.validation_percentage), len(ds) * config.dataset.validation_percentage ] #TODO: check config
        ds_train, ds_val = random_split(ds, lengths)
        trainloaders.append(DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)) #TODO: config.batch_size
        valloaders.append(DataLoader(ds_val, batch_size=config.batch_size)) #TODO: config.batch_size
    return trainloaders, valloaders