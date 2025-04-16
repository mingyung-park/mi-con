import os
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from settings import *
from models import *
from trainer import *
from inference import *
from utils import *
from dataset import *


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def get_latest_experiment_id(base_dir=EXPERIMENT_PATH, prefix="EXP"):
    existing = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    nums = [int(d.replace(f"{prefix}_", "")) for d in existing if d.replace(f"{prefix}_", "").isdigit()]
    return max(nums) + 1 if nums else 1

def run_experiment(experiment_config_file):
    # ✅ config 통합
    experiment_cfg = load_yaml(os.path.join(EXPERIMENT_CONFIG_PATH ,experiment_config_file))

    # ✅ config.yaml 파일을 읽어옵니다.    
    aug_cfg = load_yaml(os.path.join(CONFIG,experiment_cfg["augmentation"]))
    model_cfg = load_yaml(os.path.join(CONFIG,experiment_cfg["model"]))
    training_cfg = load_yaml(os.path.join(CONFIG,experiment_cfg["training"]))
    compile_cfg = load_yaml(os.path.join(CONFIG,experiment_cfg["compile"]))
    
    cfg = {
        "model": model_cfg,
        "compile": compile_cfg,
        "training": training_cfg,
        "augmentation": aug_cfg,
    }

    # ✅ 실험 저장 경로 설정
    exp_id = get_latest_experiment_id(prefix=model_cfg.get("name", "model"))
    exp_name = f"{model_cfg.get('name', 'model')}_{exp_id}"
    save_dir = os.path.join("experiments", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"✅ Experiment '{exp_name}' 시작")
    print(f"✅ Experiment config: {cfg}")

    # ✅ 데이터 로딩
    train_img, train_lbl, _ = load_dataset(TRAIN_CSV)
    test_img, _, test_ids = load_dataset(TEST_CSV)
    train_img_split, val_img_split, train_lbl_split, val_lbl_split = train_test_split(
        train_img, train_lbl, test_size=0.2, stratify=train_lbl, random_state=42
    )

    print(f"✅ 데이터 로딩 완료")
    print(f"✅ 학습 데이터: {train_img_split.shape}")
    print(f"✅ 검증 데이터: {val_img_split.shape}")
    print(f"✅ 테스트 데이터: {test_img.shape}")

    # ✅ 라벨 인코더 적용
    encoder_path = experiment_cfg["labelencoder"]
    label_encoder = None

    if encoder_path is not None and os.path.exists(encoder_path):
        label_encoder = joblib.load(encoder_path)
        train_lbl_split = label_encoder.transform(train_lbl_split)
        val_lbl_split = label_encoder.transform(val_lbl_split)
    else:
        label_encoder = LabelEncoder()
        train_lbl_split = label_encoder.fit_transform(train_lbl_split)
        val_lbl_split = label_encoder.transform(val_lbl_split)
        
        # 학습된 인코더 저장
        save_encoder_path = encoder_path if encoder_path else os.path.join(LABEL_ENCODER_PATH, 'default_le.pkl')
        joblib.dump(label_encoder, save_encoder_path)
        print(f"✅ 라벨 인코더 저장 완료: {save_encoder_path}")



    # ✅ 데이터셋 생성
    train_ds = get_dataset(train_img_split, train_lbl_split, batch_size=32, training=True)
    val_ds = get_dataset(val_img_split, val_lbl_split, batch_size=32, training=False)
    test_ds = get_dataset(test_img, labels=None, training=False, batch_size=32)

    # ✅ 모델 생성 및 컴파일
    model = get_model(cfg)
    print(f"----------MODEL SUMMARY----------\n{model.summary()}\n")
    model = compile_model(model, cfg)

    # ✅ 학습, BEST MODEL 저장
    history = train_model(model, train_ds, val_ds, cfg, save_dir=save_dir)
    print(f"------------BEST MODEL------------\n{history.best_model}\n")
    print(f"-----------BEST RESULTS-----------\n{history.best_metrics}\n")
    print(f"----------TRAINING HISTORY----------\n{history}\n")
    
    # ✅ 학습 결과 시각화
    plot_history(history, metrics=["loss", "accuracy"])

    # ✅ 히스토리 결과 저장
    save_history(
        history=history,
        file_name="history.json",
        cfg=cfg,
        save_dir=save_dir   # 이미 실험별 폴더 경로
    )

    # ✅ 추론 + 제출 파일 생성
    model= history.best_model
    run_inference(
        model=model,
        test_ds=test_ds,
        sample_path="data/sample_submission.csv",
        save_dir=save_dir,
        name="submission.csv"
    )
    print(f"✅ 제출 파일 생성 완료")
    print(f"✅ Experiment '{exp_name}' 완료")

if __name__ == "__main__":
    run_experiment()
