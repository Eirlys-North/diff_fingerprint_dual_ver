import torch
import torchvision
from torch.utils.data import Subset
import torchvision.transforms as transforms
from collections import defaultdict, Counter

def get_dataset(dataset, batch_size=256, augment=True, role="defender", get_type="loader"):
    assert dataset in ["mnist", "kmnist"], "Only 'mnist' and 'kmnist' are supported in this version."


    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    if dataset == "mnist":
        full_train = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_test
        )


        data, targets = full_train.data, full_train.targets
        indices_per_class = {i: [] for i in range(10)}
        for idx, label in enumerate(targets):
            indices_per_class[label.item()].append(idx)

        subset1_indices = []
        subset2_indices = []
        for class_indices in indices_per_class.values():
            split = len(class_indices) // 2
            subset1_indices.extend(class_indices[:split])
            subset2_indices.extend(class_indices[split:])

        subset1 = Subset(full_train, subset1_indices)
        subset2 = Subset(full_train, subset2_indices)

        trainset = subset1 if role == "defender" else subset2

    elif dataset == "kmnist":
        trainset = torchvision.datasets.KMNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.KMNIST(
            root="./data", train=False, download=True, transform=transform_test
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if get_type == "loader":
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        return trainloader, testloader
    else:
        return trainset, testset
