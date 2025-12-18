import torch
import copy

class EarlyStopping:
    """
    Early stopping to stop the training when the validation loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = None 
        self.best_model_state = None 
        self.best_epoch = 0
        self.best_train_dice = None
        self.best_train_loss = None

    def __call__(self, val_dice, val_loss, train_dice, train_loss, model, epoch):
        score = val_dice
        
        if self.best_score is None:
            self.save_best(val_dice, val_loss, train_dice, train_loss, model, epoch)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_best(val_dice, val_loss, train_dice, train_loss, model, epoch)
            self.counter = 0 

    def save_best(self, val_dice, val_loss, train_dice, train_loss, model, epoch):
        self.best_score = val_dice
        self.best_loss = val_loss
        self.best_train_dice = train_dice
        self.best_train_loss = train_loss
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.best_epoch = epoch

def calculate_dice(logits, targets, smooth=1e-6):
    """
    Computes the Dice Coefficient between logits and targets.
    """
    preds = (torch.sigmoid(logits) > 0.5).float()
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

def count_parameters(model):
    """
    Counts trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
