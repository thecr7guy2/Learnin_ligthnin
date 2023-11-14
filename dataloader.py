import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS 
import torch
from torchvision import transforms,datasets
from torch.utils.data import random_split, DataLoader


class cifar_datamodule(pl.LightningDataModule):
    def __init__(self,data_dir:str= "./data",batch_size=64) :
        super().__init__()
        self.data_dir = data_dir
        self.common_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.batch_size = batch_size


    def prepare_data(self) -> None:
        datasets.CIFAR10(self.data_dir,train=True,download=True)
        datasets.CIFAR10(self.data_dir,train=False,download=True)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            # train_dataset  = datasets.CIFAR10(self.data_dir,train=True,transform=self.common_transform)
            # self.train,self.val = random_split(train_dataset,[45000, 5000], generator=torch.Generator().manual_seed(42))

        # ###### If different transformations need to be applied for train and valid:####################
            train_dataset = datasets.CIFAR10(self.data_dir, train=True, transform=self.train_transform)
            val_dataset = datasets.CIFAR10(self.data_dir, train=True, transform=self.common_transform)  # Notice we still use train=True

            # Get the indices for the split
            train_indices, val_indices = random_split(range(50000), [45000, 5000], generator=torch.Generator().manual_seed(42))

            # Use Subset to create datasets based on indices and transformations
            self.train = torch.utils.data.Subset(train_dataset, train_indices)
            self.val = torch.utils.data.Subset(val_dataset, val_indices)
        ########################################################################################

        if stage == "test":
            self.test = datasets.CIFAR10(self.data_dir,train=False,transform=self.common_transform)
        if stage == "predict":
            self.predict = datasets.CIFAR10(self.data_dir,train=False,transform=self.common_transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train,batch_size=self.batch_size,shuffle=True,num_workers=2)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val,batch_size=self.batch_size,shuffle=False,num_workers=2)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test,batch_size=self.batch_size,shuffle=False,num_workers=4)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict,batch_size=self.batch_size,shuffle=False,num_workers=4)



# cdm = cifar_datamodule()
# cdm.prepare_data()
# cdm.setup(stage='fit')
# trainloader = cdm.train_dataloader()
# print(next(iter(trainloader))[0].shape)
