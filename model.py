import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics


class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self, model, args):

        self.args = args
        super().__init__()

        self.model = model


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, stage='val')
    
    def _common_step(self, batch, batch_idx, stage='train'):
        X, y = batch['video'], batch['label']

        output = self.model(X.permute(0, 2, 1, 3, 4)) # (8, 3, 16, 224, 224) -> (8, 16, 3, 224, 224)

        loss = F.cross_entropy(output.logits, y)
        acc = torchmetrics.functional.accuracy(output.logits, y, task="multiclass", num_classes=3)

        self.log(
            f"{stage}_loss", loss.item(), batch_size=self.args["batch_size"], on_epoch=True, prog_bar=True
        )
        self.log(
            f"{stage}_acc", acc, batch_size=self.args["batch_size"], on_epoch=True, prog_bar=True
        )
        if stage == 'train':
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args["lr"],
            #weight_decay=self.args["weight_decay"],
        )
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, self.args["max_epochs"], last_epoch=-1
        #)
        return [optimizer]#, [scheduler]