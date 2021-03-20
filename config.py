import torch

# =============== Basic Configurations ===========
TEXTURE_W = 1024
TEXTURE_H = 1024
TEXTURE_DIM = 16
USE_PYRAMID = True
VIEW_DIRECTION = True

# at beginning of the script
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============== Train Configurations ===========
DATA_DIR = 'data/'
CHECKPOINT_DIR = 'data/checkpoints'
LOG_DIR = ''
TRAIN_SET = [str(i) for i in range(1, 1200)]
EPOCH = 50
BATCH_SIZE = 12
CROP_W = 256
CROP_H = 256
LEARNING_RATE = 1e-3
BETAS = '0.9, 0.999'
L2_WEIGHT_DECAY = '0.01, 0.001, 0.0001, 0'
EPS = 1e-8
LOAD = None
LOAD_STEP = 0
EPOCH_PER_CHECKPOINT = 50


# =============== Test Configurations ============
TEST_LOAD = ''
TEST_DATA_DIR = ''
TEST_SET = [str(i) for i in range(200)]
SAVE_DIR = 'data/output'


# ============= Render Config ====================
OUT_MODE = "video"
FPS = 30
