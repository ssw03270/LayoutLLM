# LLaMA-Factory Fine-Tuning & Inference Guide

## 1. 데이터셋 적용
LLaMA-Factory에서 fine-tuning을 진행하기 위해 데이터셋을 적용하는 방법은 다음과 같습니다.

1. `LLaMA-Factory/data` 폴더 내에 `html_layout` 폴더를 생성합니다.
2. 원하는 데이터셋 파일을 `html_layout` 폴더 내에 저장합니다.
3. 데이터셋 정보는 `dataset_info.json` 파일에 작성합니다.

## 2. Fine-Tuning 진행
Fine-tuning을 진행하는 방법은 다음과 같습니다.

1. `examples/train_lora` 폴더에서 학습을 위한 YAML 파일을 작성합니다.
2. 아래 명령어를 실행하여 fine-tuning을 진행합니다.

```sh
llamafactory-cli train examples/train_lora/train_diningroom.yaml
```

3. 특정 GPU만을 사용하여 학습을 진행하려면 다음과 같이 실행합니다.

```sh
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train examples/train_lora/train_diningroom.yaml
```

## 3. Inference 실행
Fine-tuning된 모델을 API 형태로 사용할 수 있습니다. 다음과 같이 실행합니다.

```sh
API_PORT=8000 llamafactory-cli api examples/inference/test_diningroom.yaml
```

이를 통해 fine-tuning된 LLM을 GPT API처럼 사용할 수 있습니다.
예제는 `LLaMA-Factory/example/test_api/test_bedroom.py` 파일을 참고하세요.

