# 데이터셋 준비 및 전처리 가이드

## 1. 데이터셋 준비
전처리를 수행하기 전에 데이터셋을 다운로드해야 합니다. 아래 링크를 통해 **InstructScene** 데이터셋을 다운로드하세요.

[InstructScene 데이터셋 다운로드](https://huggingface.co/datasets/chenguolin/InstructScene_dataset/tree/main)

다운로드한 데이터는 `dataset` 폴더에 저장해야 합니다. 폴더 구조는 다음과 같이 구성되어야 합니다.

```
preprocessing/dataset/dataset/3D-FRONT
preprocessing/dataset/dataset/InstructScene
```

## 2. 전처리 실행
데이터셋을 전처리하려면 `preprocessing/dataset` 폴더 내의 `data_preprocessing.py`를 실행하세요.

```sh
python preprocessing/dataset/data_preprocessing.py
```

만약 데이터셋 구성 (예: prompt) 을 변경하고 싶다면 아래의 함수를 수정하면 됩니다.

```python
def generate_dataset(...):
    # 데이터셋 생성 로직 수정
```

코드가 정상적으로 실행되었다면 아래 경로에 데이터셋 파일이 생성됩니다.

```
preprocessing/dataset
```

생성된 데이터셋은 `LLaMA-Factory` 폴더 내의 README.md 파일에 안내된 대로 사용하면 됩니다.

