import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split


class DataManager():
    def __init__(self, dataset='cifar10', batch_size=32, remain_ratio=0, validation_ratio=0.1, num_clients=10, random_seed=1234):
        self.dataset = dataset
        self.trainloader, self.testloader, self.dl_clients, self.remainset = self.__load_data(
            dataset=dataset,
            batch_size=batch_size,
            remain_ratio=remain_ratio,
            validation_ratio=validation_ratio,
            num_clients=num_clients,
            random_seed=random_seed,
        )

    def get_data(self):
        return self.trainloader, self.testloader, self.dl_clients, self.remainset

    def __load_data(self, dataset='cifar10', batch_size=32, remain_ratio=0, validation_ratio=0.1, num_clients=10, random_seed=1234):

        load_func = {
            'cifar10': self.load_cifar10,
            'cifar100' : self.load_cifar100,
        }    

        trainset, testset = load_func[dataset]()

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        remain_ratio = int(remain_ratio) if remain_ratio > 1 else int(len(trainset) * remain_ratio)

        if remain_ratio > 0:
            remainset, ds_target = random_split(trainset, [remain_ratio, len(trainset) - remain_ratio], torch.Generator().manual_seed(random_seed))
            
        else:
            remainset, ds_target = None, trainset

        num_clients_data = [len(ds_target) // num_clients for _ in range(num_clients)]
        num_clients_data[-1] = len(ds_target) % num_clients if len(ds_target) % num_clients != 0 else num_clients_data[-1]

        ds_clients = random_split(ds_target, num_clients_data, torch.Generator().manual_seed(random_seed))
        dl_clients = []
        
        for ds_client in ds_clients:
            len_val = int(len(ds_client) * validation_ratio)
            len_train = len(ds_client) - len_val

            
            ds_train, ds_val = random_split(ds_client, [len_train, len_val], torch.Generator().manual_seed(random_seed))
            dl_clients.append({
                'train': DataLoader(ds_train, batch_size=batch_size, shuffle=True),
                'val': DataLoader(ds_val, batch_size=batch_size, shuffle=True),
            })

        return trainloader, testloader, dl_clients, remainset

    def load_cifar10(self):
        # Download and transform CIFAR-10 (train and test)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # TODO: make the first parameter (path) to be defined from user
        trainset = CIFAR10("./data", train=True, download=True, transform=transform_train)
        testset = CIFAR10("./data", train=False, download=True, transform=transform_test)

        return trainset, testset

    def load_cifar100(self):
        # Download and transform CIFAR-100 (train and test)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
    
        # TODO: make the first parameter (path) to be defined from user
        trainset = CIFAR100("./data", train=True, download=True, transform=transform_train)
        testset = CIFAR100("./data", train=False, download=True, transform=transform_test)

        return trainset, testset




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