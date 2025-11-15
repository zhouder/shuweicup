import os

class Config:
    # 数据路径
    DATA_DIR = '../AgriculturalDisease_trainingset'
    VAL_DIR = '../AgriculturalDisease_validationset'
    SAVE_DIR = './checkpoints'
    FIGURES_DIR = './figures'
    
    # 模型配置
    MODEL_TYPE = 'yolo11x-cls'
    NUM_CLASSES = 61
    PRETRAINED = True
    
    # 训练参数
    BATCH_SIZE = 128
    EPOCHS = 150
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05
    USE_AMP = True
    
    # 学习率调度
    WARMUP_EPOCHS = 5
    WARMUP_START_LR = 1e-6
    MIN_LR = 1e-6
    
    # 数据增强
    IMG_SIZE = 384
    CROP_SCALE = (0.7, 1.0)
    CROP_RATIO = (3/4, 4/3)
    COLOR_JITTER = {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
    GRAYSCALE_PROB = 0.1
    ERASING_PROB = 0.25
    
    # RandAugment
    RANDAUGMENT_NUM_OPS = 2
    RANDAUGMENT_MAGNITUDE = 8
    
    # Mixup/CutMix
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 0.4
    MIXUP_PROB = 0.7
    
    # EMA
    USE_EMA = True
    EMA_DECAY = 0.999
    
    # 评估指标
    SAVE_TOP_K = 3
    
    # 随机种子
    SEED = 42
    
    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)