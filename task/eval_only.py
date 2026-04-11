import torch
import config as cfg
from dataset import build_dataloaders
from model import WhaleConformer, WeightedBCEWithFocal
from train import validate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading datasets...")
    _, _, val_loader = build_dataloaders()

    print("Loading best model...")
    model = WhaleConformer(
        n_classes=cfg.n_classes(), d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF, n_layers=cfg.N_LAYERS, conv_kernel_size=cfg.CONV_KERNEL,
        dropout=cfg.DROPOUT, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH, sample_rate=cfg.SAMPLE_RATE,
    ).to(device)

    # Point this to your actual best model path!
    ckpt_path = "runs/finetune_20260410_163840/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    # Handle DataParallel prefix if it exists
    state_dict = ckpt["model_state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Use the tuned thresholds from your previous run output!
    tuned_thresholds = torch.tensor([0.8, 0.8, 0.8, 0.5, 0.8, 0.8, 0.8], device=device)
    
    print("Running validation with class-specific post-processing...")
    criterion = WeightedBCEWithFocal().to(device)  # Just a dummy for validate to work

    val_results = validate(model, val_loader, criterion, device, tuned_thresholds, epoch=60)

    print(f"\nFinal Overall Mean F1: {val_results['mean_f1']:.3f}")


if __name__ == "__main__":
    main()