# train.py

import torch
from dataset import get_dataloaders
# from ugrkan import UGRKAN
from ukan import UKAN
from loss import BCEDiceLoss
from engine import train
from utils import seed_everything,load_config
import argparse
def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description="Run a segmentation experiment from a YAML config.")
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        help="The name of the dataset configuration to use (e.g., 'busi', 'cvc_clinicdb')."
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None, # Default to None, so we know if it was provided
        help="Override the seed for reproducibility from the config file."
    )
    args = parser.parse_args()
    full_config = load_config()
    if args.dataset not in full_config['datasets']:
        raise ValueError(f"Dataset '{args.dataset}' not found in config.yaml. Available datasets: {list(full_config['datasets'].keys())}")
    config = full_config['common'] | full_config['datasets'][args.dataset]
    print(f"--- Running experiment for dataset: {args.dataset} ---")
    if args.seed is not None:
        # The --seed argument was provided, so it takes priority
        final_seed = args.seed
        config['seed'] = final_seed # Update the config dict
        print(f"Using command-line --seed override: {final_seed}")
    else:
        # No --seed argument, use the one from the config file
        final_seed = config['seed']
        print(f"Using seed from config file: {final_seed}")
    seed_everything(config['seed'])
    train_loader, val_loader = get_dataloaders(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UKAN(num_classes=config['num_classes'], img_size=config['model_params']['img_size']).to(device)
    
    # Set up optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training_params']['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=config['training_params']['patience'], verbose=True
    )
    loss_fn = BCEDiceLoss()
    
    # Start training
    trained_model, history = train(
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        device
    )
    
    print("Best Mean IoU:", max(history['mean_iou']))
    print("Dice:", max(history['dice']))
if __name__ == '__main__':
    main()