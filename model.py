import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, auc, ConfusionMatrixDisplay


class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self, model, args):

        self.args = args
        super().__init__()

        self.model = model
        self.dataloader_length = 0
        self.classes = ['Feeding', 'Grooming', 'Pumping']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, stage='val')
    
    def _common_step(self, batch, batch_idx, stage='train'):
        X, y = batch['video'], batch['label']

        output = self(X.permute(0, 2, 1, 3, 4)) # (8, 3, 16, 224, 224) -> (8, 16, 3, 224, 224)

        loss = F.cross_entropy(output.logits, y)
        acc = torchmetrics.functional.accuracy(output.logits, y, task="multiclass", num_classes=3)

        self.log(
            f"{stage}_loss", loss.item(), batch_size=self.args["batch_size"], on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            f"{stage}_acc", acc, batch_size=self.args["batch_size"], on_step=False, on_epoch=True, prog_bar=True
        )
        if stage == 'train':
            return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch['video'], batch['label']
        output = self(X.permute(0, 2, 1, 3, 4))

        # Convert the predicted probabilities to class predictions
        y_pred = torch.argmax(output.logits, dim=1)
        y_prob = torch.softmax(output.logits, dim=1)
        y_probs = y_prob.cpu().numpy()

        if batch_idx == 0:
            self.all_y = y.cpu().numpy()
            self.all_y_probs = y_probs
        else:
            self.all_y = np.concatenate([self.all_y, y.cpu().numpy()])
            self.all_y_probs = np.concatenate([self.all_y_probs, y_probs])


        # Perform the computations for the last batch
        if batch_idx == self.dataloader_length - 1:
            # Compute the ROC curve
            fprs = []
            tprs = []
            for i in range(len(self.classes)):
                fpr, tpr, threshold = roc_curve((self.all_y == i), self.all_y_probs[:, i])
                fprs.append(fpr)
                tprs.append(tpr)

            self._plot_roc_curve(fprs, tprs)

            # Generate the confusion matrix
            cm = confusion_matrix(self.all_y, np.argmax(self.all_y_probs, axis=1))

            # Plot the confusion matrix
            self._plot_confusion_matrix(cm)

            # Compute the classification report
            report = classification_report(self.all_y, np.argmax(self.all_y_probs, axis=1), target_names=self.classes)
            print(report)

        return output.logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args["lr"],
        )
        return [optimizer]

    def _plot_roc_curve(self, fprs, tprs):
        # Plot the ROC curve for each class
        plt.figure()
        for i in range(len(self.classes)):
            plt.plot(fprs[i], tprs[i], label=f'Class {self.classes[i]}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('fly_roc.png')

    def _plot_confusion_matrix(self, cm):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=self.classes)
        disp.plot()
        plt.savefig('fly_cm.png')