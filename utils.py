import os
import torch

import numpy as np

import conf as cfg

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.weights_save_dir = cfg.SAVE_WEIGHT_DIR
        
        if not os.path.exists(self.weights_save_dir):
            os.mkdir(self.weights_save_dir)
        
        self.metric_min = np.inf

    def __call__(self, target_score, model):    
        self._save_checkpoint(target_score, model, islast=True)

        if not self.best_score:
            self._save_checkpoint(target_score, model)
            self.best_score = target_score
            
        elif (target_score > self.best_score + self.delta):
            print(f"Current score: {target_score}, best score: {self.best_score}")
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self._save_checkpoint(target_score, model)
            self.counter = 0

    def _save_checkpoint(self, target_score, model, islast=False):        
        
        if not os.path.isdir(self.weights_save_dir):
            os.mkdir(self.weights_save_dir)
        
        if islast:
            torch.save(model.state_dict(), f"{self.weights_save_dir}/last_weights.pt")
            
        else:
            if self.verbose and self.best_score:
                print(f'Validation loss improved ({self.best_score:.6f} --> {target_score:.6f}).  Saving model ...')
                torch.save(model.state_dict(), f"{self.weights_save_dir}/best_weights.pt")
                self.best_score = target_score
            
            else:
                torch.save(model.state_dict(), f"{self.weights_save_dir}/best_weights.pt")
                self.best_score = target_score