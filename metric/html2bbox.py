import os
import json
import pickle

from tqdm import tqdm

from bs4 import BeautifulSoup
import re

from objfeat_vqvae.objfeat_vqvae import ObjectFeatureVQVAE

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def extract_float(value):
    match = re.search(r"[-+]?\d*\.?\d+", value)  # 숫자 부분만 추출
    return float(match.group()) if match else None

def parse_html_layout(html_content):
    """
    Parses an HTML layout file containing <rect> elements with 3D transform properties.

    Args:
        html_content (str): The HTML content as a string.

    Returns:
        list: A list of dictionaries, each representing an object with its properties.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    objects = []

    for rect in soup.find_all('rect'):
        category = rect.get("data-category", "unknown")
        image_data = rect.get("data-image", "[]")

        # Extract transform properties
        transform = rect.get("transform", "")
        translate_match = re.search(r"translate3d\(([^)]+)\)", transform)
        scale_match = re.search(r"scale3d\(([^)]+)\)", transform)
        rotate_match = re.search(r"rotateY\(([^)]+)\)", transform)

        if translate_match and scale_match:
            try:
                x, y, z = map(lambda v: float(re.search(r"[-+]?\d*\.?\d+", v).group()),
                              translate_match.group(1).split(','))
            except:
                x, y, z = 0, 0, 0
            try:
                scale_str = scale_match.group(1).strip()  # 공백 제거
                scale_str = scale_str.replace("..", ".")  # 중복된 소수점 수정
                w, h, d = map(float, scale_str.split(','))
            except:
                w, h, d = 0, 0, 0
        else:
            x, y, z, w, h, d = 0, 0, 0, 0, 0, 0

        try:
            rotation = float(rotate_match.group(1)) if rotate_match else 0
        except:
            rotation = 0

        # Extract image indices
        image_ids = re.findall(r"\[img:(\d+)\]", image_data)
        image_ids = [int(img_id) for img_id in image_ids]

        objects.append({
            "category": category,
            "x": x, "y": y, "z": z,
            "w": w, "h": h, "d": d,
            "rotation": rotation,
            "images": image_ids,
            "obj_feat": []
        })

    return objects

with open("./objfeat_vqvae/objfeat_bounds.pkl", "rb") as f:
    kwargs = pickle.load(f)
model = ObjectFeatureVQVAE(
            "openshape_vitg14",
            "gumbel",
            **kwargs
        )
import torch
from torch import LongTensor
checkpoint = torch.load("./objfeat_vqvae/threedfront_objfeat_vqvae_epoch_01999.pth", map_location=torch.device('cuda'))
model.load_state_dict(checkpoint["model"])
model.eval()

test_dataset = load_json("bedroom_prediction_file.json")
for file_idx, test_data in enumerate(tqdm(test_dataset)):
    parse_gt = parse_html_layout(test_data["ground_truth"])
    parse_pred = parse_html_layout(test_data["prediction"])

    for obj_idx in range(len(parse_gt)):
        gt_obj_features = model.reconstruct_from_indices(LongTensor([parse_gt[obj_idx]["images"]])).detach().cpu().numpy()
        parse_gt[obj_idx]["obj_feat"] = gt_obj_features.tolist()

    for obj_idx in range(len(parse_pred)):
        pred_obj_features = model.reconstruct_from_indices(LongTensor([parse_pred[obj_idx]["images"]])).detach().cpu().numpy()
        parse_pred[obj_idx]["obj_feat"] = pred_obj_features.tolist()

    output_dir = f"./output/{file_idx}/"
    os.makedirs(output_dir, exist_ok=True)
    save_json(os.path.join(output_dir, "gt.json"), parse_gt)
    save_json(os.path.join(output_dir, "pred.json"), parse_pred)
