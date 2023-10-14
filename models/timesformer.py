from .base_model import BaseModel
import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification


class Timesformer(BaseModel):
    def __init__(
        self,
        num_classes,
        num_frames,
        model_ckpt,
        image_processor_ckpt,
        label2id,
        id2label
    ):
        super(Timesformer, self).__init__(num_classes, num_frames)

        self.model = TimesformerForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
            num_frames=num_frames,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_ckpt)

    def forward(self, x):
        return self.model(x)

    def _freeze_layers(self, freeze_type):
        for param in self.model.timesformer.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self, pretrained_weights_path):
        state_dict_model = torch.load(pretrained_weights_path)['state_dict']
        for key in list(state_dict_model.keys()):
            state_dict_model[key.replace('model.', '')] = state_dict_model.pop(key)
        self.model.load_state_dict(state_dict_model)
