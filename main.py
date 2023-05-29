import os
import torch
import lightning.pytorch as pl
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from utils import create_preprocessor_config
from model import VideoClassificationLightningModule
from data_module import FlyDataModule

def main(mode = None):

    # PATH INFO
    PROJ_DIR = '/cta/users/mpekey/FlyVideo'
    TRAIN_DATA_PATH = os.path.join(PROJ_DIR, 'FlyTrainingData', 'Train')
    VAL_DATA_PATH = os.path.join(PROJ_DIR, 'FlyTrainingData', 'Validation')

    # MODEL INFO
    MODEL_CHECKPOINT = "MCG-NJU/videomae-base"

    # DATASET INFO
    class_labels = ['Feeding', 'Grooming', 'Pumping']
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_CHECKPOINT,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        num_frames = 16 # Default is 16
    )

    model_args = create_preprocessor_config(model, image_processor, sample_rate=8, fps=30)

    args = {
        # Data
        "train_data_path" : TRAIN_DATA_PATH,
        "val_data_path" : VAL_DATA_PATH,
        "lr" : 0.01,
        #"weight_decay" : 1e-4,
        "max_epochs" : 25,
        "batch_size" : 16,
        "video_path_prefix" : '',
        "video_min_short_side_scale" : 256,
        "video_max_short_side_scale" : 320,
        "clip_duration" : model_args["clip_duration"],
        "crop_size" : model_args["crop_size"],
        "num_frames_to_sample": model_args["num_frames_to_sample"],
        "video_means" : model_args["image_mean"],
        "video_stds" : model_args["image_std"]
    }

    # Freeze the model
    for param in model.videomae.parameters():
        param.requires_grad = False

    trainer = pl.Trainer(
        max_epochs=25,
        callbacks=[TQDMProgressBar(refresh_rate=args["batch_size"])],
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=40
    )

    classification_module = VideoClassificationLightningModule(model, args)
    data_module = FlyDataModule(args)

    if mode is None:
        print('Please enter a mode!')
    elif mode == 'fit':
        trainer.fit(classification_module, data_module)
    elif mode == 'test':
        trainer.test(classification_module, data_module)


if __name__ == '__main__':
    main(mode = 'fit')