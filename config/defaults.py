"""Configs."""
from fvcore.common.config import CfgNode


# -----------------------------------------------------------------------------
# Some constants.
# -----------------------------------------------------------------------------
M_PI = 3.141592653589793

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# -----------------------------------------------------------------------------
# Plot options
# -----------------------------------------------------------------------------
_C.PLOT = CfgNode()

# DPI.
_C.PLOT.DPI = 100

# Image shorter side size, in pixels.
_C.PLOT.IMG_SIZE = 1000

# Map size, in meters.
_C.PLOT.MAP_SIZE = (5000, 5000)

# Viewport size, in meters.
_C.PLOT.VIEWPORT_SIZE = 100

# -----------------------------------------------------------------------------
# BN options
# -----------------------------------------------------------------------------
_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# Data root.
_C.DATA.ROOT = "./data/waymo_open_dataset_motion_v_1_1_0"

# Meta root.
_C.DATA.META_FILE = "./csv/meta.csv"

# Number of data loader workers per training process.
_C.DATA.NUM_WORKERS = 4

# Load data to pinned host memory.
_C.DATA.PIN_MEMORY = True

# Dataset class name
_C.DATA.DATASET = "WaymoDataset"

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model name.
_C.MODEL.MODEL_NAME = "Wayformer"

# Number of output gmm modes.
_C.MODEL.OUTPUT_MODES = 64

# -----------------------------------------------------------------------------
# Solver options
# -----------------------------------------------------------------------------
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 2e-4

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 200

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "adamw"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None

# -----------------------------------------------------------------------------
# Map options
# -----------------------------------------------------------------------------
_C.MAP = CfgNode()

# Crop map size, centered at ego, in meters.
_C.MAP.MAP_CROP_SIZE = 160

# Map polyline interpolate space interval, in meters.
_C.MAP.INTERP_INTERVAL = 0.5

# -----------------------------------------------------------------------------
# Track options
# -----------------------------------------------------------------------------
_C.TRACK = CfgNode()

# History size.
_C.TRACK.HISTORY_SIZE = 10

# Future size.
_C.TRACK.FUTURE_SIZE = 80

# Track size.
_C.TRACK.SIZE = _C.TRACK.HISTORY_SIZE + _C.TRACK.FUTURE_SIZE + 1

# -----------------------------------------------------------------------------
# Meta options
# -----------------------------------------------------------------------------
_C.META = CfgNode()

# Max speed threshold for stationary, in m/s.
_C.META.STATIC_MAX_SPEED = 2

# Max displacement distance for stationary, in meters.
_C.META.STATIC_MAX_DIST = 5

# Max lateral displacement distance for straight, in meters.
_C.META.STRAIGHT_MAX_LAT_DIST = 5

# Max absolute heading diff for straight, in rad.
_C.META.STRAIGHT_MAX_HEADING_DIFF = M_PI / 6.0

# Min longitudinal displacement distance for U-turn, in meters.
_C.META.UTURN_MIN_LON_DIST = -2

# Max displacement distance for short length track, in meters.
_C.META.SHORT_MAX_DIST = 10

# Min displacement distance for long length track, in meters.
_C.META.LONG_MIN_DIST = 30

# Max displacement distance for valid long length track, in meters.
_C.META.LONG_MAX_DIST = 200

# Max agent distance to ego, in meters.
_C.META.MAX_DIST_TO_EGO = 80

# -----------------------------------------------------------------------------
# Feature options
# -----------------------------------------------------------------------------
_C.FEATURE = CfgNode()

# Number of nearby agents around target agent.
_C.FEATURE.NUM_NEARBY_AGENTS = 64

# Radius to query road graph & nearby agents, in meters.
_C.FEATURE.AGENT_NEARBY_RADIUS = 80

# Fixed interpolation interval for map polyline, in meters.
_C.FEATURE.POLYLINE_INTERP_INTERVAL = 0.5

# Max polyline length, in meters.
_C.FEATURE.MAX_POLYLINE_LEN = 30

# Max number of map elements.
_C.FEATURE.MAX_NUM_MAP_FEATS = 320

# -----------------------------------------------------------------------------
# Scenario options
# -----------------------------------------------------------------------------
_C.SCENARIO = CfgNode()

# Step size.
_C.SCENARIO.STEP_SIZE = 5

# Two consective frame timestamp diff threshod, in milliseconds.
_C.SCENARIO.MAX_TIME_DIFF = 110

# -----------------------------------------------------------------------------
# Training options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 8

# Resume from checkpoint
_C.TRAIN.RESUME = False

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint dir.
_C.TRAIN.CHECKPOINT_DIR = "./checkpoint"

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 0


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
