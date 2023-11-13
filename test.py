import torch
import lightning.pytorch as pl
from dataloader import cifar_datamodule
import lightning as L
import torchvision.models as models
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim
import torchmetrics
import wandb
import random

cdm = cifar_datamodule()
cdm.prepare_data()
cdm.setup(stage='fit')
valloader = cdm.val_dataloader()
set_val_samples = next(iter(valloader))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define your hyperparameters
hyperparams = {
    'learning_rate':1e-3,
    'weight_decay': 1e-5,
    'batch_size': 128
}

# Lets use Resnet-18 as the primary model
class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
        self.initialize_weights()
    def forward(self,x):
        return self.resnet18(x)
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# num_classes = 10 
# model = CustomResNet18(num_classes=num_classes)
# model = model.to(device)
# summary(model, input_size=(16, 3, 32, 32))

class LitCifarClassifier(L.LightningModule):
    def __init__(self,num_classes,samples) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = CustomResNet18(self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_images,self.val_labels = samples
    def training_step(self, batch, batch_idx):
        inputs,targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train_loss",loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss 
    def validation_step(self, batch, batch_idx):
        ###############################
        val_inputs,val_targets = batch
        val_outputs = self.model(val_inputs)
        val_loss = self.criterion(val_outputs,val_targets)
        preds = torch.argmax(val_outputs, dim=1)
        acc = self.accuracy(preds,val_targets)
        ###############################
        self.log("val_loss", val_loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc",acc)
        ################################
        return val_loss 
    def on_validation_epoch_end(self):
        val_images = self.val_images.to(self.device)
        val_labels = self.val_labels.to(self.device)
        outputs = self.model(val_images)
        preds = torch.argmax(outputs, dim=1)
        processed_images = [wandb.Image(img, caption=f"Pred: {pred}, Label: {label}") 
                        for img, pred, label in zip(val_images, preds,val_labels)]
        self.logger.experiment.log({"validation_data": processed_images})
        return preds
    def test_step(self, batch, batch_idx):
        test_inputs,test_targets = batch
        test_outputs = self.model(test_inputs)
        preds = torch.argmax(test_outputs, dim=1)
        acc = self.accuracy(preds,test_targets)
        self.log("test_acc",acc)
        return preds
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
        return optimizer 


model = LitCifarClassifier(num_classes=10,samples=set_val_samples)
wandb_logger = pl.loggers.WandbLogger( project="Learnin_lightnin",name="dev_run5",save_dir="runs")
wandb_logger.log_hyperparams(hyperparams)
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode="max")
trainer = pl.Trainer(max_epochs=10,logger=wandb_logger,callbacks=[checkpoint_callback])
cifardm = cifar_datamodule()
trainer.fit(model, datamodule=cifardm)
trainer.test(model,ckpt_path="best",datamodule=cifardm)


wandb.finish()



