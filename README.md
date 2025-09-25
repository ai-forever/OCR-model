# OCR model

This is a model for Optical Character Recognition based on [CRNN-arhitecture](https://arxiv.org/abs/1507.05717) and [CTC loss](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

OCR-model is a part of [ReadingPipeline](https://github.com/ai-forever/ReadingPipeline) repo.

## Demo

In the [demo](scripts/OCR-GoogleColab.ipynb) you can find an example of using of OCR-model (you can run it in your Google Colab).

## Quick setup and start

- Nvidia drivers >= 470, CUDA >= 11.4
- [Docker](https://docs.docker.com/engine/install/ubuntu/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

The provided [Dockerfile](Dockerfile) is supplied to build an image with CUDA support and cuDNN.

### Preparations

- Clone the repo.
- Download and extract dataset to the `data/` folder.
- `sudo make all` to build a docker image and create a container.
  Or `sudo make all GPUS=device=0 CPUS=10` if you want to specify gpu devices and limit CPU-resources.

If you don't want to use Docker, you can install dependencies via requirements.txt

## Configuring the model

You can change the [ocr_config.json](scripts/ocr_config.json) and set the necessary training and evaluating parameters: alphabet, image size, saving path, etc.

```
"train": {
    "datasets": [
        {
            "csv_path": "/workdir/data/dataset_1/train.csv",
            "prob": 0.5
        },
        {
            "csv_path": "/workdir/data/dataset_2/train.csv",
            "prob": 0.7
        },
        ...
    ],
    "epoch_size": 10000,
    "batch_size": 512
}
```
- `epoch_size` - the size of an epoch. If you set it to `null`, then the epoch size will be equal to the amount of samples in the all datasets.
- It is also possible to specify several datasets for the train/validation/test, setting the probabilities for each dataset separately (the sum of `prob` can be greater than 1, since normalization occurs inside the processing).

## Prepare data

Datasets must be pre-processed and have a single format: each dataset must contain a folder with images (crop images with text) and csv file with annotations. The csv file should contain two columns: "filename" with the relative path to the images (folder-name/image-name.png), and "text"-column with the image transcription.

| filename          | text |
| ----------------- | ---- |
| images/4099-0.png | is   |

If you use polygon annotations in COCO format, you can prepare a training dataset using this script:

```bash
python scripts/prepare_dataset.py \
    --annotation_json_path path/to/the/annotaions.json \
    --annotation_image_root dir/to/images/from/annotation/file \
    --class_names pupil_text pupil_comment teacher_comment \
    --bbox_scale_x 1 \
    --bbox_scale_y 1 \
    --save_dir dir/to/save/dataset \
    --output_csv_name data.csv
```

## Training

To train the model run:

```bash
python scripts/train.py --config_path path/to/the/ocr_config.json
```

## Evaluating

To test the model run:

```bash
python scripts/evaluate.py \
--config_path path/to/the/ocr_config.json \
--model_path path/to/the/model-weights.ckpt
```

If you want to use a beam search decoder with LM, you can pass lm_path arg with path to .arpa kenLM file.
--lm_path path/to/the/language-model.arpa

## ONNX

You can convert Torch model to ONNX to speed up inference on cpu.

```bash
python scripts/torch2onnx.py \
--config_path path/to/the/ocr_config.json \
--model_path path/to/the/model-weights.ckpt
```

