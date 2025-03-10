# 후처리 및 평가 가이드

## 1. 후처리 준비
Discrete code를 object feature로 변환하기 위한 모델을 다운로드해야 합니다. 아래 링크에서 해당 모델을 받을 수 있습니다.

[모델 다운로드](https://huggingface.co/datasets/chenguolin/InstructScene_dataset/tree/main)

다운로드한 모델 파일은 다음 경로에 저장해야 합니다.

```
metric/objfeat_vqvae/threedfront_objfeat_vqvae_epoch_01999.pth
```

## 2. 후처리 실행
LLaMA-Factory를 이용해 fine-tuning 및 inference를 진행했다면, 결과를 시각화하기 위해 post-processing을 수행해야 합니다.

이를 위해 다음 파일을 실행하세요.

```sh
python metric/html2bbox.py
```

파일이 정상적으로 실행되었다면, 결과물은 `metric/output` 폴더에 저장됩니다.

## 3. 평가
평가는 `metric` 폴더 내의 `compute_fid_scores.py` 파일을 실행하여 진행할 수 있습니다.

```sh
python metric/compute_fid_scores.py
```

이때, `compute_fid_scores.py` 파일에서 `gt_base_dir` 및 `pred_base_dir` 변수를 수정하여 평가 대상을 변경할 수 있습니다. 해당 변수에는 후처리 실행을 통해 생성된 결과 경로를 지정하면 됩니다.