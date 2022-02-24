import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import os
import argparse


def numbers2coords(list_of_numbers):
    """Convert list of numbers to list of tuple coords x, y."""
    bbox = [[list_of_numbers[i], list_of_numbers[i+1]]
            for i in range(0, len(list_of_numbers), 2)]
    return np.array(bbox)


def upscale_bbox(bbox, upscale_x=1, upscale_y=1):
    """Increase size of the bbox."""
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]

    y_change = (height * upscale_y) - height
    x_change = (width * upscale_x) - width

    x_min = max(0, bbox[0] - int(x_change/2))
    y_min = max(0, bbox[1] - int(y_change/2))
    x_max = bbox[2] + int(x_change/2)
    y_max = bbox[3] + int(y_change/2)
    return x_min, y_min, x_max, y_max


def polygon2bbox(polygon):
    x_min = np.inf
    y_min = np.inf
    x_max = -np.inf
    y_max = -np.inf
    for x, y in polygon:
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
    return int(x_min), int(y_min), int(x_max), int(y_max)


def img_crop(img, bbox):
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def class_names2id(class_names, data):
    """Match class names to categoty ids using annotation in COCO format."""
    category_ids = []
    for class_name in class_names:
        for category_info in data['categories']:
            if category_info['name'] == class_name:
                category_ids.append(category_info['id'])
    return category_ids


def get_data_from_image(data, image_id, class_names):
    texts = []
    bboxes = []
    category_ids = class_names2id(class_names, data)
    for idx, data_ann in enumerate(data['annotations']):
        if (
            data_ann['image_id'] == image_id
            and data_ann['category_id'] in category_ids
            and data_ann['attributes']
            and data_ann['attributes']['translation']
            and data_ann['segmentation']
        ):
            polygon = numbers2coords(data_ann['segmentation'][0])
            bbox = polygon2bbox(polygon)
            bboxes.append(bbox)
            texts.append(data_ann['attributes']['translation'])
    return texts, bboxes


def make_large_bbox_dataset(
    input_coco_json, image_root, class_names, bbox_scale_x, bbox_scale_y,
    save_dir, save_csv_name, image_folder_name='images'
):
    os.makedirs(save_dir, exist_ok=True)
    save_image_dir = os.path.join(save_dir, image_folder_name)
    os.makedirs(save_image_dir, exist_ok=True)

    with open(input_coco_json, 'r') as f:
        data = json.load(f)

    texts = []
    crop_names = []
    for data_img in tqdm(data['images']):
        img_name = data_img['file_name']
        image_id = data_img['id']
        image = cv2.imread(os.path.join(image_root, img_name))

        texts_from_image, bboxes_from_image = \
            get_data_from_image(data, image_id, class_names)

        crop_data = zip(texts_from_image, bboxes_from_image)
        for idx, (text, bbox) in enumerate(crop_data):
            upscaled_bbox = upscale_bbox(bbox, bbox_scale_x, bbox_scale_y)
            crop = img_crop(image, upscaled_bbox)
            crop_name = f'{image_folder_name}/{image_id}-{idx}.png'
            crop_path = os.path.join(save_dir, crop_name)
            cv2.imwrite(crop_path, crop)
            texts.append(text)
            crop_names.append(crop_name)

    data = pd.DataFrame(zip(crop_names, texts), columns=["filename", "text"])
    csv_path = os.path.join(save_dir, save_csv_name)
    data.to_csv(csv_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_json_path', type=str, required=True,
                        help='Path to json with segmentation dataset'
                        'annotation in COCO format.')
    parser.add_argument('--annotation_image_root', type=str, required=True,
                        help='Directory to folder with images from'
                        'annotatin.json.')
    parser.add_argument('--class_names', nargs='+', type=str, required=True,
                        help='Class namess (separated by spaces) from '
                        'annotation_json_path to make OCR dataset from them.')
    parser.add_argument('--bbox_scale_x', type=float, required=True,
                        help='Scale parameter for bbox.')
    parser.add_argument('--bbox_scale_y', type=float, required=True,
                        help='Scale parameter for bbox.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save OCR dataset.')
    parser.add_argument('--output_csv_name', type=str, required=True,
                        help='The name of the output csv with OCR annotation'
                        'informarion.')

    args = parser.parse_args()

    make_large_bbox_dataset(
        input_coco_json=args.annotation_json_path,
        image_root=args.annotation_image_root,
        class_names=args.class_names,
        bbox_scale_x=args.bbox_scale_x,
        bbox_scale_y=args.bbox_scale_y,
        save_dir=args.save_dir,
        save_csv_name=args.output_csv_name
    )
