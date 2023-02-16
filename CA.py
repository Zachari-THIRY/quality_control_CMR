import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings

from medpy.metric import binary

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AE(nn.Module):
    def __init__(self, keys, **kwargs):
        assert all([type(cls) == str for cls in keys]), 'keys parameter must be an array of Strings'
        assert len(keys) >= 2, 'At least two keys are necessary.'
        
        if not all([len(cls)<=3 for cls in keys]): warnings.warn("Using key labels with length >3 will lead to display incoherences")

        super().__init__()
        self.init_layers(kwargs["latent_size"])
        self.apply(self.weight_init)
        self.keys = keys
        self.loss_function = self.Loss(kwargs["functions"], kwargs["settling_epochs_BKGDLoss"], kwargs["settling_epochs_BKMSELoss"])
        self.metrics = self.Metrics(self.keys)
        self.optimizer = kwargs["optimizer"](
            self.parameters(),
            lr=kwargs["lr"],
            **{k:v for k,v in kwargs.items() if k in ["weight_decay", "momentum"]}
        )
        

    def init_layers(self, latent_size):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=latent_size, kernel_size=4, stride=2, padding=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1)
        )

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    class Loss():
        def __init__(self, functions, settling_epochs_BKGDLoss, settling_epochs_BKMSELoss):
            self.MSELoss = self.MSELoss()
            self.BKMSELoss = self.BKMSELoss()
            self.BKGDLoss = self.BKGDLoss()
            self.GDLoss = self.GDLoss()
            self.functions = functions
            self.settling_epochs_BKGDLoss = settling_epochs_BKGDLoss
            self.settling_epochs_BKMSELoss = settling_epochs_BKMSELoss

        class BKMSELoss:
            def __init__(self):
                self.MSELoss = nn.MSELoss()
            def __call__(self, prediction, target):
                return self.MSELoss(prediction, target)

        class MSELoss:
            def __init__(self):
                self.MSELoss = nn.MSELoss()
            def __call__(self, prediction, target):
                return self.MSELoss(prediction[:,1:], target[:,1:])

        class BKGDLoss:
            def __call__(self, prediction, target):
                intersection = torch.sum(prediction * target, dim=(1,2,3))
                cardinality = torch.sum(prediction + target, dim=(1,2,3))
                dice_score = 2. * intersection / (cardinality + 1e-6)
                return torch.mean(1 - dice_score)
      
        class GDLoss:
            def __call__(self, x, y):
                tp = torch.sum(x * y, dim=(0,2,3))
                fp = torch.sum(x * (1-y), dim=(0,2,3))
                fn = torch.sum((1-x) * y, dim=(0,2,3))
                nominator = 2*tp + 1e-06
                denominator = 2*tp + fp + fn + 1e-06
                dice_score =- (nominator / (denominator+1e-6))[1:].mean()
                return dice_score

        def __call__(self, prediction, target, epoch, validation=False):
            contributes = {f: self.__dict__[f](prediction, target) for f in self.functions}
            if "BKGDLoss" in contributes and epoch < self.settling_epochs_BKGDLoss:
                contributes["BKGDLoss"] += self.BKGDLoss(prediction[:,1:], target[:,1:])
            if "BKMSELoss" in contributes and epoch < self.settling_epochs_BKMSELoss:
                contributes["BKMSELoss"] += self.BKMSELoss(prediction[:,1:], target[:,1:])
            contributes["Total"] = sum(contributes.values())
            if validation:
                return {k: v.item() for k,v in contributes.items()}
            else:
                return contributes["Total"]

    class Metrics():
        def __init__(self, keys):
            self.DC = self.DC()
            self.HD = self.HD()
            self.keys = keys

        class DC:
            def __call__(self, prediction, target):
                try:
                    return binary.dc(prediction, target)
                except Exception:
                    return 0

        class HD:
            def __call__(self, prediction, target):
                try:
                    return binary.hd(prediction, target)
                except Exception:
                    return np.nan

        def __call__(self, prediction, target, validation=False):
            metrics = {}
            for c,key in enumerate([cls+'_' for cls in self.keys]):
                ref = np.copy(target)
                pred = np.copy(prediction)

                ref = np.where(ref!=c, 0, 1)
                pred = np.where(pred!=c , 0, 1)

                metrics[key + "dc"] = self.DC(pred, ref)
                metrics[key + "hd"] = self.HD(pred, ref)
            return metrics

    def training_routine(self, epochs, train_loader, val_loader, ckpt_folder=None):
        if ckpt_folder is not None and not os.path.isdir(ckpt_folder):
            os.mkdir(ckpt_folder)
        history = []
        best_acc = None
        for epoch in epochs:
            self.train()
            for patient in train_loader:
                for batch in patient:
                    batch = batch.to(device)
                    self.optimizer.zero_grad()
                    reconstruction = self.forward(batch)
                    loss = self.loss_function(reconstruction, batch, epoch)
                    loss.backward()
                    self.optimizer.step()
                    
            self.eval()
            with torch.no_grad():
                result = self.evaluation_routine(val_loader, epoch)
            
            if ckpt_folder is not None and (best_acc is None or result['Total'] < best_acc or epoch%10 == 0):
                ckpt = os.path.join(ckpt_folder,"{:03d}.pth".format(epoch))
                if best_acc is None or result['Total'] < best_acc:
                    best_acc = result['Total']
                    ckpt = ckpt.split(".pth")[0] + "_best.pth"
                torch.save({"AE": self.state_dict(), "AE_optim": self.optimizer.state_dict(), "epoch": epoch}, ckpt)
                clean_old_checkpoints(ckpt_folder)

            self.epoch_end(epoch, result)
            history.append(result)
        return history

    def evaluation_routine(self, val_loader, epoch):
        epoch_summary = {}
        for patient in val_loader:
            gt, reconstruction = [], []
            for batch in patient:
                batch = {"gt": batch.to(device)}
                batch["reconstruction"] = self.forward(batch["gt"])
                gt = torch.cat([gt, batch["gt"]], dim=0) if len(gt) > 0 else batch["gt"]
                reconstruction = torch.cat([reconstruction, batch["reconstruction"]], dim=0) if len(reconstruction) > 0 else batch["reconstruction"]
                for k,v in self.loss_function(batch["reconstruction"], batch["gt"], epoch, validation=True).items():
                    if k not in epoch_summary.keys():
                        epoch_summary[k]=[]
                    epoch_summary[k].append(v)
            
            gt = np.argmax(gt.cpu().numpy(), axis=1)
            gt = {"ED": gt[:len(gt)//2], "ES":gt[len(gt)//2:]}
            reconstruction = np.argmax(reconstruction.cpu().numpy(), axis=1)
            reconstruction = {"ED": reconstruction[:len(reconstruction)//2], "ES": reconstruction[len(reconstruction)//2:]}
            for phase in ["ED", "ES"]:
                for k,v in self.metrics(reconstruction[phase], gt[phase]).items():
                    if k not in epoch_summary.keys():
                        epoch_summary[k] = []
                    epoch_summary[k].append(v)
        epoch_summary = {k: np.mean(v) for k,v in epoch_summary.items()}
        return epoch_summary

    def epoch_end(self,epoch,result):
        print("\033[1mEpoch [{}]\033[0m".format(epoch))
        header, row = "", ""
        for k,v in result.items():
          header += "{:.6}\t".format(k)
          row += "{:.6}\t".format("{:.4f}".format(v))
        print(header)
        print(row)

    def load_checkpoint(self, data_path:os.PathLike='default', checkpoint_path:os.PathLike='default', eval:bool=False ):
        """
        Loads the model checkpoint, by default the last best saved.
        Setting a value for checkpoint_path will override data_path default behaviour.

        Parameters
        ----------
            data_path: os.PathLike (default = 'default')
                The root dir of the data (eg. data/brain), only useful is checkpoint_path isn't provided
            checkpoint_path: os.PathLike (default = 'default')
                If 'default' is set, then last best saved checkpoint will be loaded. Otherwise, provide path to custom checkpoint.
            eval:bool (default = False)
                If set to True, ae.eval() will also be called. Leave to False for training.
        """

        assert data_path != 'default' or checkpoint_path != 'default', "Either data_path or checkpoint_path must be provided"

        if checkpoint_path=='default':
            ckpt = os.path.join(data_path,"checkpoints/", sorted([file for file in os.listdir(os.path.join(data_path,"checkpoints")) if "_best" in file])[-1])
        else:
            ckpt = checkpoint_path
        print("Chosen checkpoint is {} .".format(os.path.split(ckpt)[1]))
        print("###################################")
        ckpt = torch.load(ckpt)

        self.load_state_dict(ckpt["AE"])
        self.optimizer.load_state_dict(ckpt["AE_optim"])
        if eval: self.eval()
    
def plot_history(history):
    losses = [x['Total'] for x in history]
    plt.plot(losses, '-x', label="loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()

#######################
#Hyperparameter Tuning#
#######################

def get_sets(parameters, set_parameters=None):
    if set_parameters is None:
        set_parameters = {k: None for k in parameters.keys()}
    if None not in set_parameters.values():
        yield set_parameters
    else:
        current_index = list(set_parameters.values()).index(None)
        current_parameter = list(set_parameters.keys())[current_index]
        for value in parameters[current_parameter]:
            set_parameters[current_parameter] = value
            loader = get_sets(parameters, set_parameters=set_parameters.copy())
            while True:
                try:
                    yield next(loader)
                except StopIteration:
                    break 

def satisfies_rules(rules, set_parameters):
    for rule in rules:
        keys = np.unique(rule.split('"')[1::2])
        for key in keys:
            if key in set_parameters:
                rule = rule.replace('"' + key + '"', 'set_parameters["' + key + '"]')
        if not eval(rule):
            return False
    return True


# TOCHANGE: is harcoded for heart key values
def hyperparameter_tuning(parameters, train_loader, val_loader, transform, transform_augmentation, rules=[], fast=False):
    best_dc = 0
    optimal_parameters = None
    for set_parameters in get_sets(parameters):
        if not satisfies_rules(rules, set_parameters):
            continue
        print(set_parameters)

        BATCH_SIZE = set_parameters["BATCH_SIZE"]
        DA = set_parameters["DA"]
        train_loader.set_batch_size(BATCH_SIZE)
        val_loader.set_batch_size(BATCH_SIZE)
        train_loader.set_transform(transform_augmentation if DA else transform)
        val_loader.set_transform(transform)

        ae = AE(**set_parameters).to(device)
        
        history = ae.training_routine(
            range(0,set_parameters["tuning_epochs"]),
            train_loader,
            val_loader
        )
        history = {k:[x[k] for x in history] for k in history[0].keys() if k in ["LV_dc", "MYO_dc", "RV_dc"]}
        history = pd.DataFrame.from_dict(history)
        
        wasBlack = any(np.all(history.values==0, axis=1))
        isNotBlack = all(history.values[-1] > 0.01)
        avg_dc = np.mean(history.values[-1])

        if wasBlack and isNotBlack:
            if avg_dc > best_dc:
                best_dc = avg_dc
                optimal_parameters = set_parameters.copy()
            if fast:
                break

    return optimal_parameters

###############
#Checkpointing#
###############


def clean_old_checkpoints(ckpt_folder:str, best_keep:int=2, total_keep:int=10):
    """
        Is called after each training routine: will keep the {best_keep} best checkpoints, and the other most recent checkpoints for a total of {total_keep checkpoints}. 
        Others will be deleted.
        Example: keep 070_best, 077_best, and 0_78, 0_78, ... and some others

        Parameters
        ----------
            ckpt_folder: str
                The string path to the checkpoints folder
            best_keep: int (default = 2)
                The number of last best checkoints to keep
            total_keep: int (default = 10)
                The total number of checkoints to keep, best and not best included
    """
    assert(best_keep <= total_keep)
    assert(os.path.isdir(ckpt_folder))
    poor_keep = total_keep - best_keep
    poor_ckpts = sorted([file for file in os.listdir(ckpt_folder) if "_best" not in file])
    best_ckpts = sorted([file for file in os.listdir(ckpt_folder) if "_best" in file])
    if len(poor_ckpts)+len(best_ckpts)>total_keep :
        if(len(best_ckpts)>best_keep):
            for file in best_ckpts[:-best_keep] :
                file_path = os.path.join(ckpt_folder, file)
                os.remove(file_path)
        if(len(poor_ckpts)>poor_keep):
            for file in poor_ckpts[:-poor_keep]:
                file_path = os.path.join(ckpt_folder, file)
                os.remove(file_path)