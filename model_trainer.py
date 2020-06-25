import time
from typing import Callable, Union, Dict, Optional
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
# from pdb import set_trace as stp
# import model_loader_utils
from . import model_loader_utils
from . import models
import utils

DISABLE_CHECKPOINTS = False

def save_checkpoint(state, is_best=True, filename: Union[str, Path] = 'output/checkpoint.pth.tar'):
    """
    Save checkpoint if a new best is achieved. This will always create the
    checkpoint directory as if it were going to create the file, but only
    actually creates and saves the file if is_best is true.

    This also prints some feedback to the standard output stream about
    if the model was saved or not.

    Parameters
    ----------
    state : dict
        the dictionary to serialize and save
    is_best : bool
        true to save the dictionary to file, false just to create the directory
    filename : str
        a path to where the checkpoint file should be saved
    """
    if DISABLE_CHECKPOINTS:
        return
    filename = Path(filename)
    filedir = filename.parents[0]
    Path.mkdir(filedir, parents=True, exist_ok=True)
    if is_best:
        print("=> Saving model to {}".format(filename))
        torch.save(state, str(filename), pickle_protocol=4)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")

class DefaultStatsTracker:
    def __init__(self, batches_per_epoch, phase: str, accuracy: bool = True):
        self.batches_per_epoch: int = batches_per_epoch
        self.accuracy: bool = accuracy
        self.running_avg_loss_tot: float = 0.
        self.running_avg_loss_epoch: float = 0.
        self.batch: int = 0
        self.epoch: int = 0
        self.running_accuracy_tot: float = 0.
        self.running_accuracy_epoch: float = 0.
        self.losses_all = []
        self.epoch_losses = []
        self.epoch_accuracies = []
        if phase == 'train':
            self.phase = 'Training'
        elif phase == 'val':
            self.phase = 'Validation'
        else:
            raise AttributeError("phase not recognized.")

    def __call__(self, stat_dict):
        batch = stat_dict['batch']
        epoch = stat_dict['epoch']
        loss = stat_dict['loss']
        self.losses_all.append(loss)
        if batch == 0:
            self.running_avg_loss_epoch = 0.
            self.running_avg_acc_epoch = 0.

        num_terms_tot = self.batches_per_epoch * epoch + (batch + 1)
        num_terms_epoch = batch + 1
        # print(n)
        self.running_avg_loss_tot = (self.running_avg_loss_tot * ((num_terms_tot - 1) / num_terms_tot)
                                     + stat_dict['loss'] / num_terms_tot)
        self.running_avg_loss_epoch = (self.running_avg_loss_epoch * ((num_terms_epoch - 1) / num_terms_epoch)
                                       + stat_dict['loss'] / num_terms_epoch)

        # print(stat_dict['loss'])
        # print(self.running_avg_loss_epoch)
        if self.accuracy:
            out_class = torch.argmax(stat_dict['outputs'], dim=1)
            accuracy = torch.mean((out_class == stat_dict['targets']).double()).item()

            self.running_accuracy_epoch = (self.running_accuracy_epoch * ((num_terms_epoch - 1) / num_terms_epoch)
                                           + accuracy / num_terms_epoch)

        if stat_dict['epoch_end']:  # We've reached the end of an epoch
            self.epoch_losses.append(self.running_avg_loss_epoch)
            self.epoch_accuracies.append(self.running_accuracy_epoch)
            # print()
            print(f"Average {self.phase} loss over this epoch: {self.running_avg_loss_epoch}")
            if self.accuracy:
                print(f"Average {self.phase} accuracy over this epoch: {self.running_accuracy_epoch}")

    def export_stats(self):
        out_stats = dict(epoch_losses=self.epoch_losses, epoch_accuracies=self.epoch_accuracies)
        return out_stats

class DefaultLearningScheduler:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.running_avg_loss_epoch: float = 0.
        self.batch: int = 0
        self.epoch: int = 0
        self.losses_all = []
        self.epoch_losses = []

    def __call__(self, stats_dict, phase):
        batch = stats_dict['batch']
        loss = stats_dict['loss']
        self.losses_all.append(loss)
        if batch == 0:
            self.running_avg_loss_epoch = 0.

        num_terms_epoch = batch + 1
        self.running_avg_loss_epoch = (self.running_avg_loss_epoch * ((num_terms_epoch - 1) / num_terms_epoch)
                                       + stats_dict['loss'] / num_terms_epoch)

        if stats_dict['epoch_end'] and phase == 'train':  # We've reached the end of an epoch
            self.epoch_losses.append(self.running_avg_loss_epoch)
            if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
                self.scheduler.step()
            else:
                self.scheduler.step(self.running_avg_loss_epoch)

def default_save_model_criterion(stat_dict):
    # return stat_dict['epoch_end'] or stat_dict['epoch'] == 0
    return stat_dict['epoch_end']

def default_stopping_criterion(stat_dict):
    return False

def train_model(model, dataloaders, device, loss_function, optimizer, starting_epoch=0, stopping_epoch=5, out_dir=None,
                load_prev=True, learning_scheduler=None, save_model_criterion=None, stopping_criterion=None,
                stats_trackers=None):
    """

    Parameters
    ----------
    model : nn.Module
        The model to be trained. NOTE: This is modified by reference!
    dataloaders : Dict[DataLoader]
        A dictionary with keys "train" and "val". The Dataset underlying the DataLoaders should return a tuple with
        first entry being the input batch and second entry being the output labels.
    device : str
        Device to use for running the network, for instance 'cpu' or 'cuda'
    loss_function : Callable[[torch.Tensor, torch.Tensor], float]
        A function that takes the model output to an input batch drawn from "dataloaders" as the first parameter
        and the corresponding output labels as the second. This looks like loss = loss_function(outputs, targets)
    optimizer : Optimizer
        An instantiation of a class that inherets from a torch Optimizer object. Used to train the model. Examples
        are instantiations of torch.optim.SGD and of torch.optim.RMSprop
    stopping_epoch : int
        The stopping epoch. The number of training samples used in an epoch is defined by the length of the training
        dataset as contained in dataloaders, which is len(dataloaders['train'])
    out_dir : Optional[str, Path]
        The output directory in which to save the model parameters through training.
    load_prev : Union[bool, int]
        If True, check for a model that's already trained in out_dir, and load the most recent epoch. If an int, load
        epoch load_prev (epoch 0 means before training starts). If False, retrain model from epoch 0. If there are
        multiple saves per epoch, this only loads the save the corresponds with the end of an epoch (for instance,
        epoch_1_save_0.pt is the save at the end of epoch 0, so beginning of epoch 1).
    stats_trackers : Union[None, Dict[Callable], str]
        Object for tracking the statistics of the model over training.
    learning_scheduler : object
        An obect that takes in a dictionary as first argument and phase as second argument. It can, for instance,
        call an instantiation of a torch scheduler object, like those found in torch.optim.lr_scheduler, based on
        the values of the items in the dictionary.
    save_model_criterion : Optional[Callable[[Dict[int, float]], bool]] = None
        An Optional Callable that takes in the statistics of the run as defined by a dictionary and returns True if the
        model should be saved. The input dictionary has keys 'training_loss', 'validation_loss', 'training_accuracy',
        'validation_accuracy', 'training_loss_batch', 'validation_loss_batch', 'training_accuracy_batch',
        'validation_accuracy_batch', 'batch', and 'epoch'. If None, the model is saved after every epoch.
    stopping_criterion : Optional[Callable[[Dict[int, float]], bool]]
        A Callable controlling early stopping. Takes in the statistics of the run as defined by a dictionary and
        returns True if training should stop. The input dictionary has keys 'training_loss', 'validation_loss',
        'training_accuracy', 'validation_accuracy', 'training_loss_batch', 'validation_loss_batch',
        'training_accuracy_batch', 'validation_accuracy_batch', 'batch', and 'epoch'.

    """
    out_dir = Path(out_dir)
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in dataloaders}
    batches_per_epoch = {x: int(dataset_sizes[x] / dataloaders[x].batch_size) for x in ['train', 'val']}
    since = time.time()

    if isinstance(load_prev, bool) and load_prev:
        # print("Loading previous model.")
        most_recent_epoch = model_loader_utils.get_max_epoch(out_dir)
        if most_recent_epoch is not False:
            starting_epoch = min(most_recent_epoch, stopping_epoch)
            model_loader_utils.load_model_from_epoch_and_dir(model, out_dir, starting_epoch)
        else:
            starting_epoch = 0
    elif isinstance(load_prev, int):
        print("Loading previous model.")
        # most_recent_save = model_loader_utils.get_max_check_num(out_dir)
        check = model_loader_utils.load_model_from_epoch_and_dir(model, out_dir, load_prev)
        if check == -1:
            starting_epoch = 0
        else:
            starting_epoch = load_prev
    else:
        starting_epoch = 0

    model.eval()

    if learning_scheduler is None:
        def learning_scheduler(stat_dict, phase):
            pass

    if save_model_criterion is None:
        save_model_criterion = default_save_model_criterion
    if stopping_criterion is None:
        stopping_criterion = default_stopping_criterion

    if stats_trackers is None:
        stats_trackers = {x: DefaultStatsTracker(batches_per_epoch[x], x, accuracy=True) for x in ['train', 'val']}
    # train_conv_check = cc.CheckConvergence(memory_length=patience_before_stopping,
    #                                        min_length=patience_before_stopping,
    #                                        tolerance=-.004)
    stat_keys = ['loss', 'batch', 'epoch', 'epoch_end', 'outputs', 'labels']

    stat_dict = {x: None for x in stat_keys}
    stat_dict['final_epoch'] = False
    if out_dir is not None:
        Path.mkdir(out_dir, exist_ok=True)

    batch_sizes = {x: dataloaders[x].batch_size for x in ['train', 'val']}

    def end_of_epoch(phase, inputs, batch_number) -> bool:
        if dataset_sizes[phase] % batch_sizes[phase] == 0 or dataloaders[phase].drop_last == True:
            if batch_number == batches_per_epoch[phase] - 1:
                return True
            else:
                return False
        else:
            if inputs.shape[0] != batch_sizes[phase]:
                return True
            else:
                return False

    def train(epoch):
        save_ctr = 0
        stat_dict['epoch'] = epoch
        phase = 'train'
        # datalen = len(dataloaders[phase])
        model.train()
        for batch_num, (inputs, targets) in enumerate(dataloaders[phase]):
            # print(batch_num)
            # print(batch_num, " out of ", datalen, " batches")
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            model.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            try:
                loss_val = loss.item()
            except AttributeError:
                loss_val = loss
            stat_dict['loss'] = loss_val
            stat_dict['batch'] = batch_num
            stat_dict['outputs'] = outputs
            stat_dict['targets'] = targets
            stat_dict['checkpoint'] = save_model_criterion(stat_dict)

            if stat_dict['checkpoint'] and out_dir is not None:
                filename = out_dir/f'epoch_{epoch}_save_{save_ctr}.pt'
                save_checkpoint({'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'learning_scheduler_state_dict': learning_scheduler.scheduler.state_dict()},
                                filename=filename)
                save_ctr += 1

            learning_scheduler(stat_dict, phase)
            stats_trackers[phase](stat_dict)

            if loss_val > 0:
                loss.backward()
                optimizer.step()

            stat_dict['epoch_end'] = False
            del loss
            torch.cuda.empty_cache()

    def validate(epoch):
        print()
        print('Validation')
        print('-' * 10)
        stat_dict['epoch'] = epoch
        phase = 'val'
        model.eval()
        with torch.no_grad():
            for batch_num, (inputs, targets) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # forward
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss_val = loss.item()

                learning_scheduler(stat_dict, phase)
                stats_trackers[phase](stat_dict)

    for epoch in range(starting_epoch, stopping_epoch):
        tic = time.time()
        print()
        print(f'Epoch {epoch}/{stopping_epoch-1}')
        print('-' * 10)
        stat_dict['epoch_end'] = True
        train(epoch)
        validate(epoch)
        # Todo: save and stop criterion here, taking into account results of validate
        toc = time.time()
        print(f'Elapsed time this epoch: {round(toc - tic, 1)} seconds')

    if starting_epoch < stopping_epoch or stopping_epoch == 0:
        filename = out_dir/f'epoch_{stopping_epoch}_save_{0}.pt'  # End of the last epoch
        save_checkpoint({'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'learning_scheduler_state_dict': learning_scheduler.scheduler.state_dict()},
                        filename=filename)

        #
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    else:
        print('Training previously complete -- loading save from disk')
    # if classify:
    #     print('Final val Acc: {:4f}'.format(best_acc))
    # export_dict = {x: stats_trackers[x].export_stats() for x in ['train', 'val']}
    # training_log_and_machinery = dict(stats_history=export_dict, stats_trackers=stats_trackers,
    #                                   learning_scheduler=learning_scheduler, optimizer=optimizer)

    # return models_to_return, training_log_and_machinery
    # return training_log_and_machinery
    return None
