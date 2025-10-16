# engine.py

import torch
from tqdm import tqdm
from utils import AverageMeter, plot_result
from metrics import iou_score, dice_coefficient

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Performs one full training epoch."""
    model.train()
    loss_meter = AverageMeter()
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), images.size(0))
        
    return loss_meter.avg

def evaluate(config, model, dataloader, loss_fn, device, epoch):
    """Performs validation, calculates metrics, and plots results."""
    model.eval()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Validation")):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss_meter.update(loss.item(), images.size(0))

            iou, dice, _ = iou_score(outputs, labels)
            iou_meter.update(iou, images.size(0))

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            dice_meter.update(dice, images.size(0))

            if i == 0:
                plot_result(config, images[0].cpu(), outputs[0].cpu(), labels[0].cpu())

    return loss_meter.avg, iou_meter.avg, dice_meter.avg

def train(config, model, train_loader, val_loader, optimizer, scheduler, loss_fn, device):
    """Main training loop with early stopping."""
    history = {'train_loss': [], 'val_loss': [], 'mean_iou': [], 'dice': []}
    best_mean_iou = float("-inf")
    current_patience = 0

    for epoch in range(config['training_params']['num_epochs']):
        print(f"--- Epoch {epoch + 1}/{config['training_params']['num_epochs']} ---")
        
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        avg_val_loss, mean_iou, dice = evaluate(config, model, val_loader, loss_fn, device, epoch)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MeanIoU: {mean_iou:.4f} | Dice Coeff: {dice:.4f}")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['mean_iou'].append(mean_iou)
        history['dice'].append(dice)

        scheduler.step(avg_val_loss)
        
        if mean_iou > best_mean_iou:
            print("Validation MeanIoU increased. Saving model...")
            best_mean_iou = mean_iou
            current_patience = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            current_patience += 1
            if current_patience >= config['training_params']['patience']:
                print(f"Early stopping triggered after {config['training_params']['patience']} epochs.")
                break
        
        torch.cuda.empty_cache()

    print("Training finished.")
    return model, history