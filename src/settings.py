from pathlib import Path

BASE_PATH = Path("../mi-con")

# 데이터 경로
DATA_PATH = BASE_PATH / 'data'
TRAIN_CSV = DATA_PATH / 'train.csv'
TEST_CSV = DATA_PATH / 'test.csv'

# config결호
CONFIG = BASE_PATH / 'configs'

# 모델 경로
MODEL_PATH = BASE_PATH / 'models'

# 실험 경로, 로그
EXPERIMENT_PATH = BASE_PATH / 'experiments'
EXPERIMENT_CONFIG_PATH = EXPERIMENT_PATH / 'configs'

# 제출 폴더더 경로

SUBMISSION_PATH = EXPERIMENT_PATH /'submissions'

# 라벨 인코더 경로
LABEL_ENCODER_PATH = BASE_PATH / 'label_encoder'