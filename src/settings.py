from pathlib import Path

BASE_PATH = Path("../mi-con")

# 데이터 경로
DATA_PATH = BASE_PATH / 'data'
TRAIN_CSV = DATA_PATH / 'train.csv'
TEST_CSV = DATA_PATH / 'test.csv'
SUB_CSV = DATA_PATH / 'sample_submission.csv'

# config 경로
CONFIG = BASE_PATH / 'configs'

# 모델 경로
MODEL_PATH = BASE_PATH / 'models'

# 실험 경로, 로그
EXPERIMENT_PATH = BASE_PATH / 'experiments'
EXPERIMENT_LOG_PATH = EXPERIMENT_PATH / 'logs'

# 제출 폴더 경로
SUBMISSION_PATH = EXPERIMENT_PATH /'submissions'

