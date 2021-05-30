import numpy as np
from facenet_pytorch import fixed_image_standardization
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from torchvision import datasets, transforms


def prepare_datasets(casia_dir, batch_size):
    transformation = transforms.Compose([
        transforms.Resize(160), 
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(casia_dir, transform=transformation)

    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    train_idxs = indexes[:int(0.9 * len(indexes))]        
    test_idxs = indexes[int(0.9 * len(indexes)):]

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idxs)
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_idxs)
    )
    vizualization_with_masks = True
    visualization_cnt = 20
    indexes_visualization = np.array([i for i in range(len(dataset)) if dataset.imgs[i][1] < visualization_cnt and (vizualization_with_masks or "m." not in dataset.imgs[i][0])])  
    visualization_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indexes_visualization)
    )

    return train_loader, test_loader, visualization_loader
