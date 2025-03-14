# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import json

from tqdm import tqdm

from openai import OpenAI
from transformers.utils.versions import require_version

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"파일이 성공적으로 저장되었습니다: {file_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a bedroom-style layout using OpenAI API")
    parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY", "0"), help="API key for OpenAI")
    parser.add_argument("--api_port", type=int, default=int(os.environ.get("API_PORT", 8000)), help="API server port")
    parser.add_argument("--model", type=str, default="test", help="Model name to use for generation")
    parser.add_argument("--data_dir_path", type=str, default="data/html_layout/")
    parser.add_argument("--room_type", type=str, default="bedroom")
    return parser.parse_args()

def main():
    args = parse_arguments()

    client = OpenAI(
        api_key=args.api_key,
        base_url="http://localhost:{}/v1".format(args.api_port),
    )

    dataset_path = os.path.join(args.data_dir_path, args.room_type + "_val_dataset.json")
    test_dataset = load_json(dataset_path)

    output_list = []
    for test_data in tqdm(test_dataset):
        messages = [
            {
                "role": "user",
                "content": test_data["instruction"]
            }
        ]

        result = client.chat.completions.create(messages=messages, model=args.model)
        messages.append(result.choices[0].message)

        prediction = result.choices[0].message.content
        ground_truth = test_data["output"]

        output = {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "tag": test_data["tag"],
        }
        output_list.append(output)

    save_json(f"./{args.room_type}_prediction_file.json", output_list)

if __name__ == "__main__":
    main()
