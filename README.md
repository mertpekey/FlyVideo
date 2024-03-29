# Fruit Fly Video Classification

From the fruit fly video clips, the model predicts 3 fly behaviours; Feeding, Pumping and Grooming.

## Model Overview

The model used in this project is Timesformer. It's designed for video classification tasks. The training parameters include:
- `sample_rate`: 16
- `learning_rate`: 1e-3
- `num_frames`: 8
- Video frame size: 224x224
- Model input shape: `(batch_size, num_frames, channel, height, width)`

## Installation

You can install the required libraries using the following pip commands:

```bash
pip install lightning torch transformers torchvision wandb pytorchvideo
```

## Running Inference

To run inference using the provided `inference.py` file, follow these steps:

1. Clone the project repository:

    ```bash
    git clone https://github.com/mertpekey/FruitFlyVideo.git
    ```

2. Create a directory named `inference_data`, and inside that directory, create a `test` folder. Place the videos you want to make inferences on inside the `test` folder. You can modify the folder structure and image names if needed.

3. Create a directory named `pretrained_weights`, and download the pretrained model inside that folder. Change the name of the file to `pretrained_model.ckpt`. Or you can define it using `--model_name` argument.

4. Run the inference script with the following command:

    ```bash
    python inference.py --inference_data_path 'inference_data' --batch_size 1 --device cuda --load_ckpt
    ```

   - `--inference_data_path`: Path to the `inference_data` folder.
   - `--batch_size`: (Optional) Set the batch size for inference (default: 1).
   - `--device`: (Optional) Set the device to run inference on (e.g., `cuda` or `cpu`).
   - `--load_ckpt`: (Optional) Use this flag to indicate using a finetuned model.
   - `--pretrained_model_path`: (Optional) Use this to determine which pretrained checkpoints will be used (default: pretrained_model.ckpt).

4. The inference script will generate a `prediction.json` output file in the project directory, containing the inference results.

## Prediction Output

The `predictions.json` output file provides the results of the inference process for the input videos. Each entry in the file corresponds to a video and includes the following information:

- **Video Name**: The name of the video file for which predictions were made.

- **Clip Index**: A list of indices corresponding to the clips within the video that were used for inference. If multiple clips were used from the same video, they will have different indices.

- **Predictions**: A list of predicted classes for each clip. These classes indicate the action or content present in the corresponding clip.

- **Probabilities**: A list of lists containing class probabilities for each clip. The class probabilities are represented in the order: "Feeding", "Grooming", "Pumping". Each sublist contains three values, where the first value corresponds to the probability of the "Feeding" class, the second value to the "Grooming" class, and the third value to the "Pumping" class.

Here's an example of what a `predictions.json` file may look like:

```json
{
    "v_feeding_g01_100.avi": {
        "clip_index": [0, 1],
        "prediction": ["Feeding", "Feeding"],
        "probs": [
            [0.4673, 0.1260, 0.4066],
            [0.8415, 0.0460, 0.1126]
        ]
    }
}
```

---