from data_module import FlyDataModule
from arguments import parse_arguments
from utils import get_model, generate_class_mappings, initialize_trainer
from model import VideoClassificationLightningModule


def main(mode, load_model, args):
    # Dataset
    data_dir = "path/to/FlyTrainingData"  ## change args or config data dir
    class_labels, label2id, id2label = generate_class_mappings(data_dir)

    # Model
    #### Add get model arguments
    fly_model = get_model(
        model_type="dummy",
        num_classes=len(class_labels),
        num_frames=args.num_frames,
        image_processor_ckpt="dummy",
        model_ckpt="dummy",
        label2id=label2id,
        id2label=id2label,
    )
    if load_model:
        pretrained_weights_path = "tb_logs/timesformer_logs_s16_noES_b16_lr1e3/version_0/checkpoints/epoch=24-step=1000.ckpt"
        fly_model._load_pretrained_weights(pretrained_weights_path)
    fly_model._freeze_layers(freeze_type="dummy")

    # Model specific arguments
    fly_model_args = fly_model._create_preprocessor_config(
        sample_rate=args.sample_rate, fps=args.fps
    )
    for key, value in fly_model_args.items():
        setattr(args, key, value)

    lightning_module = VideoClassificationLightningModule(fly_model, args)
    data_module = FlyDataModule(args)

    # Lightning Trainer
    trainer = initialize_trainer(args)

    if mode is None:
        print("Please enter a mode!")
    elif mode == "fit":
        trainer.fit(lightning_module, data_module)
    elif mode == "test":
        trainer.test(lightning_module, data_module)
    elif mode == "predict":
        data_module.setup()
        lightning_module.dataloader_length = len(data_module.val_dataloader())
        print(lightning_module.dataloader_length)
        trainer.predict(lightning_module, data_module.val_dataloader())


if __name__ == "__main__":
    args = parse_arguments()
    main(mode="fit", load_model=False, args=args)
