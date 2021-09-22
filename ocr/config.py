import json

CONFIG_PATH = '/workdir/ocr/config.json'


class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get(self, key):
        return self.config[key]

    def get_train(self, key):
        return self.config['train'][key]

    def get_val(self, key):
        return self.config['val'][key]

    def get_image(self, key):
        return self.config['image'][key]

    def get_train_datasets(self, key):
        return [data[key] for data in self.config['train']['datasets']]

    def get_val_datasets(self, key):
        return [data[key] for data in self.config['val']['datasets']]


CONFIG = Config(config_path=CONFIG_PATH)
