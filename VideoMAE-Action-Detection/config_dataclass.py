from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ROIActionHeadCfg:
    FEATURE_EXTRACTOR: str = "2MLPFeatureExtractor"
    POOLER_TYPE: str = 'align3d'  # Can be 'pooling3d' or 'align3d'
    POOLER_RESOLUTION: int = 7
    POOLER_SCALE: float = 0.0625
    POOLER_SAMPLING_RATIO: int = 0  # Only used for align3d
    MEAN_BEFORE_POOLER: bool = True
    MLP_HEAD_DIM: int = 1024
    PREDICTOR: str = "FCPredictor"
    DROPOUT_RATE: float = 0.2
    NUM_CLASSES: int = 80
    PROPOSAL_PER_CLIP: int = 100
    INIT_SCALE: float = 0.001


@dataclass
class CycleACRCfg:
    ACTIVE: bool = True
    DEPTH: int = 1
    INSTANCE_DEPTH: int = 1
    LOCAL_FEATURE_LEN: int = 50
    TEMP_FEATURE_LEN: int = 8
    MEM_ACTIVE: bool = True
    MAX_PER_SEC: int = 5
    MAX_PERSON: int = 20
    DIM_IN: int = 1024
    DIM_INNER: int = 1024
    DIM_OUT: int = 1024
    PENALTY: bool = True
    LENGTH: Tuple[int, int] = (30, 30)
    MEMORY_RATE: int = 1
    FUSION: str = "add"
    CONV_INIT_STD: float = 0.01
    DROPOUT: float = 0.2
    NO_BIAS: bool = False
    LAYER_NORM: bool = True
    TEMPORAL_POSITION: bool = True
    USE_ZERO_INIT_CONV: bool = True


@dataclass
class TestCfg:
    EXTEND_SCALE: Tuple[float, float] = (-0.05, -0.1)
    BOX_THRESH: float = 0.8
    ACTION_THRESH: float = 0.0


@dataclass
class ModelCfg:
    ROI_ACTION_HEAD: ROIActionHeadCfg = ROIActionHeadCfg()
    CYCLE_ACR: CycleACRCfg = CycleACRCfg()


@dataclass
class Config:
    MODEL: ModelCfg = ModelCfg()
    TEST: TestCfg = TestCfg()


cfg = Config()
