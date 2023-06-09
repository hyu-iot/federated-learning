import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split

class DataManager():
    def __init__(self, dataset='cifar10', batch_size=32, num_pretrain=0, num_clients=10, num_federated_data=[1]):
        self.dataset = dataset
        self.trainloader, self.testloader, self.preloader, self.dl_clients = self.load_data(
            dataset=dataset,
            batch_size=batch_size,
            num_pretrain=num_pretrain,
            num_clients=num_clients,
            num_federated_data=num_federated_data
        )

    def get_FL_data(self, total):
        for dl_FL in self.dl_clients:
            if dl_FL['total'] == total:
                return dl_FL['train'], dl_FL['val']

        return [], []

    def load_data(self, dataset='cifar10', batch_size=32, num_pretrain=0, num_clients=10, num_federated_data = [1]):

        load_func = {
            'cifar10': self.load_cifar10,
            'cifar100' : self.load_cifar100,
        }    

        trainset, testset = load_func[dataset]()

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        num_pretrain = int(num_pretrain) if num_pretrain > 1 else int(len(trainset) * num_pretrain)

        if num_pretrain > 0:
            ds_pre, ds_remain = random_split(trainset, [num_pretrain, len(trainset) - num_pretrain], torch.Generator().manual_seed(42))
            preloader = DataLoader(ds_pre, batch_size=batch_size, shuffle=True)
        else:
            ds_pre, ds_remain = None, trainset
            preloader = None
        

        dl_clients = [{'total': num_fl_data, 'train':[], 'val':[]} for num_fl_data in num_federated_data]
        for i, num_fl_data in enumerate(num_federated_data):
            num_data = int(num_fl_data) if num_fl_data > 1 else int(num_fl_data * len(ds_remain))
            dl_clients[i]['total'] = num_data
            ds_client, _ = random_split(ds_remain, [num_data, len(ds_remain) - num_data], torch.Generator().manual_seed(42))

            # Split training set into `num_clients` partitions to simulate different local datasets
            partition_size = len(ds_client) // num_clients
            lengths = [partition_size] * num_clients
            datasets = random_split(ds_client, lengths, torch.Generator().manual_seed(42))

            # Split each partition into train/val and create DataLoader
            dl_trains = []
            dl_vals = []
            
            for ds in datasets:
                len_val = len(ds) // 10  # 10 % validation set
                len_train = len(ds) - len_val
                lengths = [len_train, len_val]
                ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
                dl_clients[i]['train'].append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
                dl_clients[i]['val'].append(DataLoader(ds_val, batch_size=batch_size))

        print(f"Trainset: {len(trainset)}, Testset: {len(testset)}")
        print(f"Trainloader: {len(trainloader)}, Testloader: {len(testloader)}")
        return trainloader, testloader, preloader, dl_clients


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

        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform_train)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform_test)

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
    
        trainset = CIFAR100("./dataset", train=True, download=True, transform=transform_train)
        testset = CIFAR100("./dataset", train=False, download=True, transform=transform_test)

        return trainset, testset
