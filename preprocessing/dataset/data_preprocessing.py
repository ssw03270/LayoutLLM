import os

import matplotlib.pyplot as plt
from PIL import Image

from shapely.geometry import Polygon, box
from shapely.affinity import translate

from math import cos, sin

import pickle
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from data import get_raw_dataset, filter_function
import json

from tqdm import tqdm
from typing import *

import nltk
from nltk.corpus import cmudict
nltk.download('cmudict')

def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()


def get_article(word):
    word = word.split(" ")[0]
    article = "an" if starts_with_vowel_sound(word) else "a"
    return article

def reverse_rel(rel: str) -> str:
    return {
        "above": "below",
        "below": "above",
        "in front of": "behind",
        "behind": "in front of",
        "left of": "right of",
        "right of": "left of",
        "closely in front of": "closely behind",
        "closely behind": "closely in front of",
        "closely left of": "closely right of",
        "closely right of": "closely left of"
    }[rel]

def convert_to_past_participle(verb: str) -> str:
    """동사를 과거 분사 형태로 변환하는 간단한 함수."""
    irregular_verbs = {
        "Place": "placed",
        "Put": "put",
        "Position": "positioned",
        "Arrange": "arranged",
        "Add": "added",
        "Set up": "set up",
        "Hang": "hung",
        "Install": "installed"
    }
    return irregular_verbs.get(verb, verb + "ed")  # 기본적으로 -ed를 붙임

def fill_templates(
    desc: Dict[str, List],
    object_types: List[str], predicate_types: List[str],
    object_descs: Optional[List[str]]=None,
    seed: Optional[int]=None,
    return_obj_ids=False
) -> Tuple[str, Dict[int, int], List[Tuple[int, int, int]], List[Tuple[str, str]]]:
    if object_descs is None:
        assert object_types is not None

    if seed is not None:
        np.random.seed(seed)

    obj_class_ids = desc["obj_class_ids"]  # map from object index to class id

    # Describe the relations between the main objects and others
    selected_relation_indices = np.random.choice(
        len(desc["obj_relations"]),
        min(np.random.choice([1, 2]), len(desc["obj_relations"])),  # select 1 or 2 relations
        replace=False
    )
    selected_relations = [desc["obj_relations"][idx] for idx in selected_relation_indices]
    selected_relations = [
        (int(obj_class_ids[s]), int(p), int(obj_class_ids[o]))
        for s, p, o in selected_relations
    ]  # e.g., [(4, 2, 18), ...]; 4, 18 are class ids; 2 is predicate id
    selected_descs = []
    selected_sentences = []
    selected_object_ids = []  # e.g., [0, ...]; 0 is object id
    for idx in selected_relation_indices:
        s, p, o = desc["obj_relations"][idx]
        s, p, o = int(s), int(p), int(o)
        if object_descs is None:
            s_name = object_types[obj_class_ids[s]].replace("_", " ")
            o_name = object_types[obj_class_ids[o]].replace("_", " ")
            p_str = predicate_types[p]
            if np.random.rand() > 0.5:
                subject = f"{get_article(s_name).replace('a', 'A')} {s_name}"
                predicate = f" is {p_str} "
                object = f"{get_article(o_name)} {o_name}."
            else:  # 50% of the time to reverse the order
                subject = f"{get_article(o_name).replace('a', 'A')} {o_name}"
                predicate = f" is {reverse_rel(p_str)} "
                object = f"{get_article(s_name)} {s_name}."
        else:
            if np.random.rand() < 0.75:
                s_name = object_descs[s]
            else:  # 25% of the time to use the object type as the description
                s_name = object_types[obj_class_ids[s]].replace("_", " ")
                s_name = f"{get_article(s_name)} {s_name}"  # "a" or "an" is added
            if np.random.rand() < 0.75:
                o_name = object_descs[o]
            else:
                o_name = object_types[obj_class_ids[o]].replace("_", " ")
                o_name = f"{get_article(o_name)} {o_name}"

            p_str = predicate_types[p]
            rev_p_str = reverse_rel(p_str)

            if p_str in ["left of", "right of"]:
                if np.random.rand() < 0.5:
                    p_str = "to the " + p_str
                    rev_p_str = "to the " + rev_p_str
            elif p_str in ["closely left of", "closely right of"]:
                if np.random.rand() < 0.25:
                    p_str = "closely to the " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "closely to the " + rev_p_str.split(" ")[-2] + " of"
                elif np.random.rand() < 0.5:
                    p_str = "to the close " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "to the close " + rev_p_str.split(" ")[-2] + " of"
                elif np.random.rand() < 0.75:
                    p_str = "to the near " + p_str.split(" ")[-2] + " of"
                    rev_p_str = "to the near " + rev_p_str.split(" ")[-2] + " of"

            if np.random.rand() < 0.5:
                verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                if "lamp" in s_name:
                    verbs += ["Hang", "Install"]
                verb = verbs[np.random.choice(len(verbs))]
                subject = f"{verb} {s_name}"
                predicate = f" {p_str} "
                object = f"{o_name}."
                selected_descs.append((s_name, o_name))
                selected_object_ids.append(s)
            else:  # 50% of the time to reverse the order
                verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                if "lamp" in o_name:
                    verbs += ["Hang", "Install"]
                verb = verbs[np.random.choice(len(verbs))]
                subject = f"{verb} {o_name}"
                predicate = f" {rev_p_str} "
                object = f"{s_name}."
                selected_descs.append((o_name, s_name))
                selected_object_ids.append(o)
        selected_sentences.append(subject + predicate + object)

    text = ""
    conjunctions = [" Then, ", " Next, ", " Additionally, ", " Finnally, ", " And ", " "]
    for i, sentence in enumerate(selected_sentences):
        if i == 0:
            text += sentence
        else:
            conjunction = conjunctions[np.random.choice(len(conjunctions))]
            while conjunction == " Finnally, " and i != len(selected_sentences)-1:
                # "Finally" should be used only in the last sentence
                conjunction = conjunctions[np.random.choice(len(conjunctions))]
            if conjunction != " ":
                sentence = sentence[0].lower() + sentence[1:]
            text += conjunction + sentence

    if return_obj_ids:
        return text, selected_relations, selected_descs, selected_object_ids
    else:
        return text, selected_relations, selected_descs  # return `selected_relations`, `selected_descs` for evaluation


def predicate_types():
    return [
        "above", "left of", "in front of",
        "closely left of", "closely in front of",
        "below", "right of", "behind",
        "closely right of", "closely behind"
    ]


def save_messages(messages, output_file):
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(messages, json_file, ensure_ascii=False, indent=4)

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

def generate_dataset(raw_data, room_type, task_type="remaining values", output_path="./dataset", split="train"):
    user_prompt = ("I want to generate layout in {Domain} style. "
                   "Please generate the layout according to the following text condition."
                   "\"{TextInstruction}\". ")
    code_template = """```html
<html>
    <body>
{code}
    </body>
</html>
```"""
    assistant_prompt = """```html
<html>
    <body>
{code}
    </body>
</html>
```"""
    messages = []

    models_info_paths = raw_data._path_to_models_info
    description_paths = raw_data._path_to_descriptions
    class_labels_book = raw_data._class_labels

    text_class_labels = raw_data._class_labels
    min_bounds, max_bounds = raw_data._sizes
    minx, maxx, minz, maxz = (min_bounds[0], max_bounds[0],
                              min_bounds[2], max_bounds[2])
    max_range = max(maxx - minx, maxz - minz)
    view_bound = (-max_range, max_range, -max_range, max_range)

    min_translation = raw_data._centroids[0]
    for data_idx, data in enumerate(tqdm(raw_data)):
        data_path = data.image_path.replace("InstructScene", "topview_instruction/save_dir").replace("rendered_scene_256.png", "")
        tag = data_path.replace("dataset/topview_instruction/save_dir/", "")

        base_save_path = os.path.join(output_path, split, tag)
        os.makedirs(base_save_path, exist_ok=True)

        class_labels = data.class_labels
        translations = data.translations
        sizes = data.sizes
        angles = data.angles
        captions = data.captions

        ###### model infos
        models_info_path = models_info_paths[data_idx]
        with open(models_info_path, "rb") as f:
            data_models_info = pickle.load(f)

        description_path = description_paths[data_idx]
        with open(description_path, 'rb') as file:
            description_pkl = pickle.load(file)
        description = {"obj_class_ids": description_pkl["obj_class_ids"],
                       "obj_relations": description_pkl["obj_relations"]}

        gpt_captions = data.captions

        ###### text instruction generation
        text_instruction, _, _ = fill_templates(
            desc=description,
            object_types=class_labels_book,
            predicate_types=predicate_types(),
            object_descs=gpt_captions,
            seed=None
        )

        ###### text generation
        element_count = len(class_labels)
        gt_layout_text = ""
        masked_layout_text = ""

        text_class_index_labels = []
        text_class_index_dict = {}
        for element_idx in range(element_count):
            class_label = class_labels[element_idx]
            text_class_label = text_class_labels[np.argmax(class_label)]

            if text_class_label not in text_class_index_dict:
                text_class_index_dict[text_class_label] = 1
            else:
                text_class_index_dict[text_class_label] += 1

            text_class_index_labels.append(text_class_label + "_" + str(text_class_index_dict[text_class_label]))

        for element_idx in range(element_count):
            text_class_label = text_class_index_labels[element_idx]
            trans = translations[element_idx] - min_translation
            size = sizes[element_idx]
            angle = np.rad2deg(angles[element_idx][0])
            caption = captions[element_idx]

            data_model_info = data_models_info[element_idx]
            objfeat_vq_indices = data_model_info["objfeat_vq_indices"]
            text_objfeat = ""
            for vq_index in objfeat_vq_indices:
                text_objfeat += f"[img:{vq_index}]"

            gt_element_text = (f"<rect data-category=\"{text_class_label}\" "
                               f"data-image={text_objfeat} "
                               f"transform=\"translate3d({trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f}) "
                               f"scale3d({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}) "
                               f"rotateY({angle:.2f})\"/>")
            masked_element_text = (f"<rect data-category={text_class_label} "
                                   f"data-image=[img:FILL_idx][img:FILL_idx][img:FILL_idx][img:FILL_idx] "
                                   f"transform=\"translate3d(<FILL_x>, <FILL_y>, <FILL_z>) "
                                   f"scale3d(<FILL_x>, <FILL_y>, <FILL_z>) "
                                   f"rotateY(<FILL_deg>)\"/>")
            ##### desc
            constraints = ""
            for relation in description["obj_relations"]:
                s, p, o = relation
                if element_idx == s:
                    relation_index = p
                    target_index = o

                    relation_text = predicate_types()[relation_index]
                    relation_text = reverse_rel(relation_text)
                    text_target_class_label = text_class_index_labels[target_index]
                    constraint = f"        <constraint type=\"{relation_text}\" source=\"{text_class_label}\" target=\"{text_target_class_label}\"/> \n"
                    if constraint not in constraints:
                        constraints += constraint

                elif element_idx == o:
                    relation_index = p
                    target_index = s

                    relation_text = predicate_types()[relation_index]
                    text_target_class_label = text_class_index_labels[target_index]
                    constraint = f"        <constraint type=\"{relation_text}\" source=\"{text_class_label}\" target=\"{text_target_class_label}\"/> \n"
                    if constraint not in constraints:
                        constraints += constraint

            gt_layout_text += f"{constraints}"
            gt_layout_text += f"        {gt_element_text}"
            masked_layout_text += f"{constraints}"
            masked_layout_text += f"        {masked_element_text}"

            if element_idx < element_count - 1:
                gt_layout_text += "\n"
                masked_layout_text += "\n"

        _user_prompt = user_prompt.format(Domain=room_type, TextInstruction=text_instruction)
        _code_template = code_template.format(code=masked_layout_text)
        _user_prompt += "\n" + _code_template

        _assistant_prompt = assistant_prompt.format(code=gt_layout_text)

        message = {"instruction": _user_prompt, "input": "", "output": _assistant_prompt, "tag": tag}
        messages.append(message)

        with open(os.path.join(base_save_path, message_path), 'w', encoding='utf-8') as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    return messages

def main():
    room_types = ["bedroom", "diningroom", "livingroom"]
    for room_type in room_types:
        config_file = f"./configs/{room_type}_sg2sc_diffusion_objfeat.yaml"
        config = load_config(config_file)

        train_raw = get_raw_dataset(
            config["data"],
            filter_function(
                config["data"],
                split=config["training"].get("splits", ["train", "val"])
            ),
            path_to_bounds=None,
            split=config["training"].get("splits", ["train", "val"]),
        )
        train_messages = generate_dataset(train_raw, room_type, output_path="./dataset", split="train")

        val_raw = get_raw_dataset(
            config["data"],
            filter_function(
                config["data"],
                split=config["validation"].get("splits", ["test"])
            ),
            path_to_bounds=None,
            split=config["validation"].get("splits", ["test"])
        )
        val_messages = generate_dataset(val_raw, room_type, output_path="./dataset", split="test")

        save_messages(train_messages, f"{room_type}_train_dataset.json")
        save_messages(val_messages, f"{room_type}_val_dataset.json")

if __name__ == "__main__":
    main()


