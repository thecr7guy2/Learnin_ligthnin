from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
import lightning.pytorch as pl
from dataloader import cifar_datamodule
import lightning as L
import torchvision.models as models
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim
from pytorch_lightning.loggers import WandbLogger


# cdm = cifar_datamodule()
# cdm.prepare_data()
# cdm.setup(stage='fit')
# trainloader = cdm.train_dataloader()
# print(next(iter(trainloader))[0].shape)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Lets use Resnet-18 as the primary model
class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
    def forward(self,x):
        return self.resnet18(x)

# num_classes = 10 
# model = CustomResNet18(num_classes=num_classes)
# model = model.to(device)
# summary(model, input_size=(16, 3, 32, 32))

class LitCifarClassifier(L.LightningModule):
    def __init__(self,num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = CustomResNet18(self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
    def training_step(self, batch, batch_idx):
        inputs,targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train_loss",loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        val_inputs,val_targets = batch
        val_outputs = self.model(val_inputs)
        val_loss = self.criterion(val_outputs,val_targets)
        self.log("val_loss", val_loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.SGD(self.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        return optimizer    


model = LitCifarClassifier(num_classes=10)
wandb_logger = WandbLogger( project="Learnin_lightnin",name="dev_run")
trainer = pl.Trainer(max_epochs=10,logger=wandb_logger)
cifardm = cifar_datamodule()
trainer.fit(model, datamodule=cifardm)




