from easydict import EasyDict as edict
cfg = edict()

# data settings
cfg.TRAIN_DATA_FILE = r'your_path/train_patients.json'  
cfg.EVAL_DATA_FILE = r'your_path/val_patients.json'
cfg.TEST_DATA_FILE = r'your_path/test_patients.json'

# model settings
cfg.INPUT_SHAPE = [224, 224]  # this value may be changed in train_cae.py

cfg.EPOCHS = 100
cfg.LR = 0.001
cfg.DECAY_STEPS = 100000
cfg.DECAY_RATE = 0.95
cfg.STEPS_PER_EPOCH = 100
cfg.AMP = False
cfg.WARMUP=2  # first 2 epoch using ce loss, otherwise use focal loss
cfg.CHECKPOINTS_ROOT = 'checkpoints'
cfg.MAX_KEEPS_CHECKPOINTS = 1
cfg.EARLY_BREAK = 20



