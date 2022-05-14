import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, savepath, n_classes=1, CBAM=False):
        score = -val_loss
        if len(model) == 1:

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint_1(val_loss, model[0], savepath)
            elif score < self.best_score + self.delta:
                self.counter += 1
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint_1(val_loss, model[0], savepath)
                self.counter = 0

        elif len(model) == 3:
            model_name = ['model_GAM', 'GAM_att', 'GAM_fc']
            if self.best_score is None:
                self.best_score = score
                for i, m in enumerate(model):
                    name = model_name[i]
                    self.save_checkpoint_2(val_loss, m, savepath, name)
            elif score < self.best_score + self.delta:
                self.counter += 1
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                for i, m in enumerate(model):
                    name = model_name[i]
                    self.save_checkpoint_2(val_loss, m, savepath, name)
                self.counter = 0

        elif len(model) > 2:
            if n_classes == 1:
                if CBAM:
                    model_name = ['model1_GAM', 'model2_GAM', 'model3_GAM', 'GAM_fc', 'GAM_att']
                else:
                    model_name = ['model1', 'model2', 'model3', 'fc', 'att', 'GAM']
            elif n_classes == 2:
                if CBAM:
                    model_name = ['model1_GAM_cla', 'model2_GAM_cla', 'model3_GAM_cla', 'GAM_fc_cla', 'GAM_att_cla']
                else:
                    model_name = ['model1_cla', 'model2_cla', 'model3_cla', 'fc_cla', 'att_cla']

            if self.best_score is None:
                self.best_score = score
                for i, m in enumerate(model):
                    name = model_name[i]
                    self.save_checkpoint_many(val_loss, m, savepath, name)
            elif score < self.best_score + self.delta:
                self.counter += 1
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                for i, m in enumerate(model):
                    name = model_name[i]
                    self.save_checkpoint_many(val_loss, m, savepath, name)
                self.counter = 0


    def save_checkpoint_1(self, val_loss, model, savepath):
        '''Saves model when validation loss decrease.'''
        save = savepath + 'model_BCNN_dCor_age_NKI' + '.pt'
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), save)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

    def save_checkpoint_2(self, val_loss, model, savepath, name):
        '''Saves model when validation loss decrease.'''
        save = savepath + name + '.pt'
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), save)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

    def save_checkpoint_many(self, val_loss, model, savepath, name):
        '''Saves model when validation loss decrease.'''

        save = savepath + name + '_age.pt'
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), save)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss
