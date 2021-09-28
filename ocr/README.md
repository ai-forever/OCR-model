# OCR model

### Configuring the model

You can change the [config.json](config.json) (or make a copy of the file) and set the necessary training and evaluating parameters: alphabet, image size, saving path and etc.

It is also possible to specify several datasets for the train, validation and test, setting the probabilities for each dataset separately (the sum of probabilities can be greater than 1, since normalization occurs inside the processing). For example:

```json
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

### Training

To train the model:

```bash
python ocr/train.py --config_path path/to/the/config.json
```

### Evaluating

To test the model:

```bash
python ocr/evaluate.py \
--config_path path/to/the/config.json \
--model_path path/to/the/model-weights.ckpt
```
