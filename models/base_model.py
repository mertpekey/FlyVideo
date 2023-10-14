import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, num_classes, num_frames, **kwargs):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames

    def _freeze_layers(self, freeze_type):
        raise NotImplementedError

    def _load_pretrained_weights(self, pretrained_weights_path):
        raise NotImplementedError

    def _create_preprocessor_config(self, sample_rate=8, fps=30):
        mean = self.image_processor.image_mean
        std = self.image_processor.image_std

        if "shortest_edge" in self.image_processor.size:
            height = width = self.image_processor.size["shortest_edge"]
        else:
            height = self.image_processor.size["height"]
            width = self.image_processor.size["width"]

        crop_size = (height, width)

        num_frames_to_sample = self.num_frames
        clip_duration = num_frames_to_sample * sample_rate / fps
        print("Clip Duration:", clip_duration, "seconds")

        return {
            "video_means": mean,
            "video_stds": std,
            "crop_size": crop_size,
            "num_frames_to_sample": num_frames_to_sample,
            "clip_duration": clip_duration,
        }
