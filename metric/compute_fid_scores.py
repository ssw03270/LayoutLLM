from cleanfid import fid
import os
import shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# FID score: 118.482249
# CLIP-FID score: 6.639561
# KID score: 0.001279
def main():
    # 원본 데이터 폴더 (현재 작업 경로)
    gt_base_dir = "./diningroom_outputs/ours/layout_output/threed_front_diningroom"  # 실제 폴더 경로에 맞게 수정
    # pred_base_dir = "./livingroom_outputs/ours/layout_output/threed_front_livingroom"  # ours

    # pred_base_dir = "./outputs/instruct_scene/epoch_01999"  # instruct_scene
    pred_base_dir = "./diningroom_outputs/layoutgpt/image_output"  # layoutgpt

    # 임시 FID 평가 폴더
    temp_real = os.path.join(gt_base_dir, "temp_real")
    temp_syn = os.path.join(pred_base_dir, "temp_syn")

    # 기존 폴더 삭제 후 새로 생성
    for folder in [temp_real, temp_syn]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Ground Truth 및 Prediction 폴더 정리
    gt_dirs = sorted([d for d in os.listdir(gt_base_dir) if d.endswith("_gt")])
    #pred_dirs = sorted([d for d in os.listdir(pred_base_dir) if d.endswith("_pred")])  # ours
    pred_dirs = sorted([d for d in os.listdir(pred_base_dir)])

    # 이미지 복사 함수 (폴더 이름을 포함하여 복사)
    def copy_images(src_dirs, base_dir, dest_dir, suffix):
        for src in src_dirs:
            src_path = os.path.join(base_dir, src)
            if os.path.isdir(src_path):
                for img_file in os.listdir(src_path):
                    if img_file.endswith((".png", ".jpg", ".jpeg")):  # 이미지 파일만 복사
                        src_img = os.path.join(src_path, img_file)
                        dest_img = os.path.join(dest_dir, f"{src}_{img_file}")  # 폴더명_파일명 으로 저장
                        shutil.copy(src_img, dest_img)

    # gt 폴더에서 이미지 복사
    copy_images(gt_dirs, gt_base_dir, temp_real, "_gt")

    # pred 폴더에서 이미지 복사
    copy_images(pred_dirs, pred_base_dir, temp_syn, "_pred")

    # Compute FID/KID scores
    print("Compute FID/KID scores\n")
    eval_info = ""
    configs = {
        "fdir1": temp_real,
        "fdir2": temp_syn,
        "device": "cuda",
        "num_workers": 0  # 멀티프로세싱 비활성화
    }

    fid_score = fid.compute_fid(**configs)
    print(f"FID score: {fid_score:.6f}\n")
    eval_info += f"FID score: {fid_score:.6f}\n"
    clip_fid_score = fid.compute_fid(model_name="clip_vit_b_32", **configs)
    print(f"CLIP-FID score: {clip_fid_score:.6f}\n")
    eval_info += f"CLIP-FID score: {clip_fid_score:.6f}\n"
    kid_score = fid.compute_kid(**configs)
    print(f"KID score: {kid_score:.6f}\n")
    eval_info += f"KID score: {kid_score:.6f}\n"

    print("\n" + eval_info)


if __name__ == "__main__":
    main()