import torch
import torch.nn as nn
from tqdm import tqdm
from time import sleep

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, x, y, mask_ind):
        sq = (x - y) **2
        masked_sq = sq * (~mask_ind.type(torch.bool))
        return masked_sq.mean()
    
    
class Trainer:
    def __init__(self, model, crit, optimizer, scheduler):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def train(self, train_loader):
        self.model.train()
        total_train_loss = 0
        with tqdm(train_loader, unit = "batch") as tepoch:
            for x_c, x_n, _, mask_ind in tepoch:
                _x_n = self.model(x_n)
                train_loss =  self.crit(_x_n, x_c, mask_ind)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                total_train_loss += train_loss.item()
                tepoch.set_postfix(train_loss = train_loss.item())
                sleep(0.01)
            return total_train_loss
        
    def validate(self,valid_loader):
        self.model.eval()
        with torch.no_grad():
            total_val_loss = 0
            with tqdm(valid_loader, unit = "batch") as tepoch:
                for x_c, x_n, _, mask_ind in tepoch:
                    _x_n = self.model(x_n)
                    val_loss = self.crit(_x_n, x_c, mask_ind).item()
                    total_val_loss += val_loss
                    tepoch.set_postfix(valid_loss = val_loss)
        return total_val_loss


