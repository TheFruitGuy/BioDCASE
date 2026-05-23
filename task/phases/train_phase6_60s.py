import config as cfg
cfg.COLLAR_MIN_S = 15.0
cfg.COLLAR_MAX_S = 30.0
cfg.EVAL_SEGMENT_S = 60.0
cfg.EVAL_OVERLAP_S = 4.0
#cfg.BATCH_SIZE = 16

from train import main
main()