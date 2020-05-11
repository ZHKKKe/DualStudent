import torchvision.transforms as transforms

from third_party.mean_teacher import data
from third_party.mean_teacher import datasets as mt_dataset
from third_party.mean_teacher.utils import export
from third_party.fastswa import datasets as fswa_dataset


@export
def cifar10(tnum=2):
    dataset = mt_dataset.cifar10()

    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    dataset['train_transformation'] = data.TransformNTimes(
        transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]), n=tnum)
    
    dataset['datadir'] = 'third_party/' + dataset['datadir']
    return dataset


@export
def cifar100(tnum=2):
    dataset = fswa_dataset.cifar100()

    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    dataset['train_transformation'] = data.TransformNTimes(
        transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]), n=tnum)

    dataset['datadir'] = 'third_party/' + dataset['datadir']
    return dataset


@export
def usps(tnum=2):
    channel_stats = dict(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
    train_transformation = data.TransformNTimes(transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]), n=tnum)
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'third_party/data-local/images/usps',
        'num_classes': 10,
    }


@export
def mnist(tnum=2):
    channel_stats = dict(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
    train_transformation = data.TransformNTimes(transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]), n=tnum)
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'third_party/data-local/images/mnist',
        'num_classes': 10,
    }
