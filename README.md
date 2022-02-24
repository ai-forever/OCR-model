# OCR CRNN and CTC loss model

## Quick setup and start

- Nvidia drivers >= 470, CUDA >= 11.4
- [Docker](https://docs.docker.com/engine/install/ubuntu/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

The provided [Dockerfile](Dockerfile) is supplied to build an image with CUDA support and cuDNN.

Also you can install the necessary python packages via [requirements.txt](requirements.txt)

### Preparations

- Clone the repo.
- Download and extract dataset to the `data/` folder.
- `sudo make all` to build a docker image and create a container.
  Or `sudo make all GPUS=device=0 CPUS=10` if you want to specify gpu devices and limit CPU-resources.

## Configuring the model

You can change the [ocr_config.json](scripts/ocr_config.json) (or make a copy of the file) and set the necessary training and evaluating parameters: alphabet, image size, saving path and etc.

You can set the "epoch_size" to null to train/test on the entire dataset.

It is also possible to specify several datasets for the train, validation and test, setting the probabilities for each dataset separately (the sum of probabilities can be greater than 1, since normalization occurs inside the processing). For example:

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
    ...
}
```
Datasets must be pre-processed and have a single format: each dataset must contain a folder with text images and csv file with annotations. The csv file should consist of two columns: "filename" with the relative path to the images (folder-name/image-name.png), and "text"-column with the image transcription.

## Prepare data

If you use polygon annotations in COCO format, you can crop images with text from full images to prepare a training dataset using this script:

```bash
python scripts/prepare_dataset.py \
    --annotation_json_path path/to/the/annotaions.json \
    --annotation_image_root dir/to/images/from/annotation/file \
    --class_names pupil_text pupil_comment teacher_comment \
    --bbox_scale_x 1.2 \
    --bbox_scale_y 1.3 \
    --save_dir dir/to/save/dataset \
    --output_csv_name data.csv
```

## Training

To train the model:

```bash
python scripts/train.py --config_path path/to/the/ocr_config.json
```

## Evaluating

To test the model:

```bash
python scripts/evaluate.py \
--config_path path/to/the/ocr_config.json \
--model_path path/to/the/model-weights.ckpt
```
