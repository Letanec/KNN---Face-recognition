from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset 
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import calendar 
import time 

# directories for face recognition
data_dir_train = 'datasets\casia_with_masks'                # train data folder path
data_dir_eval = 'datasets\lfw_with_masks'                   # eval data folder path

'''       
# directories for voice recognition
data_dir_train = 'spectros/train'               # train data folder path
data_dir_dev = 'spectros/dev'                   # dev data folder path
data_dir_eval = 'spectros/eval'                 # eval data folder path
'''

batch_size = 52                         # batch size
epochs = 15                             # number of epochs, 10 for pretrained, 20 for not pretrained

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transformation = transforms.Compose([
    transforms.Resize(160),
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

dataset_train = datasets.ImageFolder(data_dir_train, transform=transformation)
dataset_eval = datasets.ImageFolder(data_dir_eval, transform=transformation)

indexes = np.arange(len(dataset_train))
np.random.shuffle(indexes)
train_idxs = indexes[:int(0.8 * len(indexes))]          # split the train:test data 9:1
test_idxs = indexes[int(0.8 * len(indexes)):]

train_loader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_idxs)
)

test_loader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(test_idxs)
)

eval_loader = DataLoader(
    dataset_eval
)

# add parameter pretrained='casia-webface' for pretrained resnet 
resnet = InceptionResnetV1(classify=True, pretrained='casia-webface', num_classes=len(dataset_train.class_to_idx)).to(device)

optimizer = optim.Adam(resnet.parameters(), lr=0.001)  
scheduler = MultiStepLR(optimizer, [4000, 8000, 12000])                     # lr will be devided by 10 on those epochs

loss = torch.nn.CrossEntropyLoss()
metrics = {
    'acc': training.accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

# Train and test run
for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))

    resnet.train()
    training.pass_epoch(
        resnet, loss, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    with torch.no_grad():
        training.pass_epoch(
            resnet, loss, test_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )
    model_path = "./models/InceptionResnetV1_pretrained_" + "_time_" + str(calendar.timegm(time.gmtime()))
    torch.save(resnet.state_dict(), model_path)
writer.close()
