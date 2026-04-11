import yaml
import torch
from ultralytics import YOLO
import os
import wandb

# 1. Disable Comet so YOLO ignores it completely
os.environ["COMET_MODE"] = "disabled"

# 2. Set up Weights & Biases
os.environ["WANDB_PROJECT"] = "BioDCASE_Task2_Baseline"
os.environ['WANDB_API_KEY'] = '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'


def run():
    YAML_FILE = './custom.yaml'
    run_name = 'biodcase_baseline'  # Change to the name of your run

    # Check if CUDA is available
    print('CUDA device count:')
    print(torch.cuda.device_count())

    # Read the config file
    with open(YAML_FILE, 'r') as file:
        config = yaml.safe_load(file)

    # 3. Explicitly start W&B and print confirmation!
    try:
        wandb.init(project=os.environ["WANDB_PROJECT"], name=run_name)
        print("\nWeights & Biases initialized successfully! Live logging is active.\n")
    except Exception as e:
        print(f"\nFailed to initialize W&B: {e}\n")

    # Load a model
    model = YOLO('yolo11s.pt')

    # train the model
    best_params = {
        'iou': 0.3,
        'imgsz': 640,
        'hsv_s': 0,
        'hsv_v': 0,
        'degrees': 0,
        'translate': 0,
        'scale': 0,
        'shear': 0,
        'perspective': 0,
        'flipud': 0,
        'fliplr': 0,
        'bgr': 0,
        'mosaic': 0,
        'mixup': 0,
        'mixup': 0,
        'copy_paste': 0,
        'erasing': 0,
        'crop_fraction': 0,
    }

    # YOLO will automatically detect the active W&B run we started above
    model.train(epochs=40, batch=96, data=YAML_FILE,
                project=config['path'] + '/runs/' + run_name, resume=False, patience=0, workers=32, **best_params)

    # 4. Cleanly close the W&B run when training is done
    wandb.finish()


if __name__ == '__main__':
    run()