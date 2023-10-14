import os

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from models.timesformer import Timesformer
from models.cnn_lstm import CnnLstm


def get_model(
    model_type,
    num_classes,
    num_frames,
    image_processor_ckpt,
    model_ckpt=None,
    label2id=None,
    id2label=None,
):
    if model_type == "timesformer":
        return Timesformer(
            num_classes,
            num_frames,
            model_ckpt,
            image_processor_ckpt,
            label2id,
            id2label,
        )
    elif model_type == "cnn_lstm":
        return CnnLstm(num_classes, num_frames, image_processor_ckpt)
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def generate_class_mappings(data_dir):
    class_folders = sorted(os.listdir(data_dir))
    class_labels = class_folders
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return class_labels, label2id, id2label


def get_training_args(args):
    wandb_logger = WandbLogger(project="timesformer-wandb")
    tqdm_callback = TQDMProgressBar(refresh_rate=args.batch_size)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="max",
        dirpath="./wandb_checkpoints",
        filename=f"timesformer_b{args.batch_size}_lr{args.lr}",
    )

    return {
        "max_epochs": args.max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "auto",
        "devices": 1 if torch.cuda.is_available() else None,
        "logger": [wandb_logger],
        "callbacks": [tqdm_callback, checkpoint_callback],
    }


def initialize_trainer(args):
    training_args = get_training_args(args)

    trainer = pl.Trainer(
        max_epochs=training_args["max_epochs"],
        logger=training_args["logger"],
        callbacks=training_args["callbacks"],
        accelerator=training_args["accelerator"],
        devices=training_args["devices"],
        log_every_n_steps=40,
    )
    return trainer
