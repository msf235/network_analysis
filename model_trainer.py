import torch
from torch import nn
from pathlib import Path
import time
from torch.utils import data  # import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
import typing
from typing import Callable, Union, Dict, Optional
import copy
from pathlib import Path

import models
import tasks

DISABLE_CHECKPOINTS = False

def save_checkpoint(state, is_best=True, filename='output/checkpoint.pth.tar'):
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

 # if len(outputs.shape) == 3:

#                         if outputs.shape[-1] == 1:
#                             preds = (outputs[:, -1] < 0.5).long() * 0 + (outputs[:, -1] >= 0.5).long() * 1
#                         else:
#                             _, preds = torch.max(outputs[:, -1], -1)


# else:
# if outputs.shape[-1] == 1:
#     preds = (outputs < 0.5).long() * 0 + (outputs >= 0.5).long() * 1
# else:
#     _, preds = torch.max(outputs, -1)


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
        self.epoch: int = 0
        self.batch: int = 0
        self.losses_all = []
        self.epoch_losses = []
        if phase == 'train':
            self.phase = 'Training'
        elif phase == 'val':
            self.phase = 'Validation'
        else:
            raise AttributeError("phase not recognized.")

    def __call__(self, stats_dict):
        batch = stats_dict['batch']
        epoch = stats_dict['epoch']
        loss = stats_dict['loss']
        self.losses_all.append(loss)
        if batch == 0:
            self.running_avg_loss_epoch = 0.
            self.running_accuracy_epoch = 0.

        num_terms_tot = self.batches_per_epoch * epoch + (batch+1)
        num_terms_epoch = batch+1
        # print(n)
        self.running_avg_loss_tot = (self.running_avg_loss_tot * ((num_terms_tot - 1) / num_terms_tot)
                                     +stats_dict['loss'] / num_terms_tot)
        self.running_avg_loss_epoch = (self.running_avg_loss_epoch * ((num_terms_epoch - 1) / num_terms_epoch)
                                       + stats_dict['loss'] / num_terms_epoch)

        print(stats_dict['loss'])
        print(self.running_avg_loss_epoch)
        if self.accuracy:
            out_class = torch.argmax(stats_dict['outputs'], dim=1)
            accuracy = torch.mean((out_class == stats_dict['labels']).double())

            self.running_accuracy_epoch = (self.running_accuracy_epoch * ((num_terms_epoch - 1) / num_terms_epoch)
                                           + accuracy / num_terms_epoch)

        if stats_dict['epoch_end']: # We've reached the end of an epoch
            self.epoch_losses.append(self.running_avg_loss_epoch)
            print(f"Average {self.phase} loss over this epoch: {self.running_avg_loss_epoch}")
            if self.accuracy:
                print(f"Average {self.phase} accuracy over this epoch: {self.running_accuracy_epoch}")

    def export_stats(self):
        out_stats = dict(epoch_losses=self.epoch_losses)
        return out_stats



def train_model(model: nn.Module,
                dataloaders: Dict[str, DataLoader],
                loss_function: Callable[[torch.Tensor, torch.Tensor], float],
                optimizer: Optimizer,
                scheduler: object = None,
                stopping_epoch: int = 5,
                out_dir: Optional[str] = None,
                epochs_per_save: int = 1,
                stopping_criterion: Optional[Callable[[Dict[int, float]], bool]] = None,
                stat_trackers: Union[None, Callable, str] = None):
    """

    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    dataloaders : Dict[DataLoader]
        A dictionary with keys "train" and "val". The Dataset underlying the DataLoaders should return a tuple with
        first entry being the input batch and second entry being the output labels.
    loss_function : Callable[[torch.Tensor, torch.Tensor], float]
        A function that takes the model output and to an input batch drawn from "dataloaders" as the first parameter
        and the corresponding output labels as the second.
    optimizer : Optimizer
        An instantiation of a class that inherets from a torch Optimizer object and used to train the model. Examples
        are instantiations of torch.optim.SGD and of torch.optim.RMSprop.
    scheduler : object
        An instantiation of a torch scheduler object, like those found in torch.optim.lr_scheduler.
    stopping_epoch : int
        The maximum number of epochs used to train. An epoch size is defined by the length of the training dataset as
        contained in dataloaders: "len(dataloaders['train'])"
    out_dir : Optional[str]
        The output directory in which to save the model parameters through training.
    stats_tracker
    track_over_batches : bool = False
        If True, track the statistics of the run over batches. This influences the output dictionary of train_model.
        If True, the keys 'training_loss_batch', 'validation_loss_batch', 'training_accuracy_batch', and
        'validation_accuracy_batch' will be keys in the output dictionary, and if False these keys will not be there.
    statistic_printing_function : Optional[List[str], Callable[[Dict[int, float]], None]] = None
        An Optional parameter that can be a list of strings or a Callable. If None, 'epoch', 'training_loss',
        'validation_loss', 'training_accuracy', and 'validation_accuracy' are printed after every epoch. If a list of
        strings, it determines which of the following are printed: 'epoch', 'batch', 'training_loss',
        'validation_loss', 'training_accuracy', 'validation_accuracy', 'training_loss_batch',
        'validation_loss_batch', 'training_accuracy_batch', and 'validation_accuracy_batch'. Here the computation of
        the accuracy assumes that the argmax of the output represents the model's choice of label. If a Callable,
        it takes in the statistics of the run during training after every batch and epoch as defined by a dictionary
        with the keys listed above and prints to the console.
    return_model_criterion : Optional[List[int]] = None
        An Optional Callable that takes in the statistics of the run during training as defined by a dictionary and
        returns True if the model should be added to the list of models returned by train_model. If None,
        this list consists of the model at the furthest point in training before training is stopped.
    save_model_criterion : Optional[Callable[[Dict[int, float]], bool]] = None
        An Optional Callable that takes in the statistics of the run as defined by a dictionary and returns True if the
        model should be saved. The input dictionary has keys 'training_loss', 'validation_loss', 'training_accuracy',
        'validation_accuracy', 'training_loss_batch', 'validation_loss_batch', 'training_accuracy_batch',
        'validation_accuracy_batch', 'batch', and 'epoch'. If None, the model is saved after every epoch.
    stopping_criterion : Optional[Callable[[Dict[int, float]], bool]]
        A Callable that takes in the statistics of the run as defined by a dictionary and returns True if training
        should stop. The input dictionary has keys 'training_loss', 'validation_loss', 'training_accuracy',
        'validation_accuracy', 'training_loss_batch', 'validation_loss_batch', 'training_accuracy_batch',
        'validation_accuracy_batch', 'batch', and 'epoch'. See network_analysis.utils.stopping_criteria for example
        Callables that can be used.

    Returns
    -------
    List[nn.Module]
        The trained models at the epochs specified by return_model_criterion.
    Dict
        A dictionary that holds the statistics of the run after every epoch. May also hold statistics after every
        batch if track_over_batches is True.

    """
    out_dir = Path(out_dir)
    device = torch.device("cpu")
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in dataloaders}
    batches_per_epoch = {x: int(dataset_sizes[x] / dataloaders[x].batch_size) for x in ['train', 'val']}

    model.eval()
    print('optimizer is ', optimizer)
    stats_trackers = {x: DefaultStatsTracker(batches_per_epoch[x], x, accuracy=True) for x in ['train', 'val']}
    # train_conv_check = cc.CheckConvergence(memory_length=patience_before_stopping,
    #                                        min_length=patience_before_stopping,
    #                                        tolerance=-.004)
    stat_keys = ['loss', 'batch', 'epoch', 'epoch_end']

    stat_dict = {x: None for x in stat_keys}
    stat_dict['model'] = model
    stat_dict['dataloaders'] = dataloaders
    if out_dir is not None:
        Path.mkdir(out_dir, exist_ok=True)

    batch_sizes = {x: dataloaders[x].batch_size for x in ['train', 'val']}

    save_cnt = 0
    def save_model_criterion(stat_dict):
        start_bool = stat_dict['epoch'] == 0 and stat_dict['batch'] == 0
        return stat_dict['epoch_end'] or start_bool

    def stopping_criterion(stat_dict):
        return False

    for epoch in range(stopping_epoch + 1):  # epoch 0 corresponds with model before training
        print()
        print('Epoch {}/{}'.format(epoch, stopping_epoch))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # if scheduler is not None:
                #   scheduler.step(epoch_losses['val'][-1])  #
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            batch = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()

                # forward
                # track history only if in train and epoch > 0
                with torch.set_grad_enabled(phase == 'train' and epoch > 0):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)

                    if hasattr(model, 'get_regularizer_loss'):
                        reg_loss = model.get_regularizer_loss(inputs)
                        if reg_loss is not None:
                            loss = loss + reg_loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if epoch > 0:
                            loss.backward()
                            optimizer.step()
                        if save_model_criterion(stat_dict):
                            if out_dir is not None:
                                save_checkpoint({'state_dict': model.state_dict()},
                                                filename=out_dir/'check_{}'.format(save_cnt))
                                save_cnt = save_cnt + 1

                    loss = loss.item()
                    stat_dict['loss'] = loss
                    stat_dict['epoch'] = epoch
                    stat_dict['batch'] = batch
                    stat_dict['outputs'] = outputs
                    stat_dict['labels'] = labels
                    if dataset_sizes[phase] % batch_sizes[phase] == 0 or dataloaders[phase].drop_last == True:
                        if batch == batches_per_epoch[phase] - 1:
                            stat_dict['epoch_end'] = True
                        else:
                            stat_dict['epoch_end'] = False
                    else:
                        if inputs.shape[0] != batch_sizes[phase]:
                            stat_dict['epoch_end'] = True
                        else:
                            stat_dict['epoch_end'] = False

                    stats_trackers[phase](stat_dict)
                batch = batch + 1



                # if classify:
                #     if len(outputs.shape) == 3:
                #         running_corrects += torch.sum(preds == labels[:, -1].long())
                #     else:
                #         running_corrects += torch.sum(preds == labels.long())


    #         if phase == 'val':
    #             stop_bool = train_conv_check.check_increasing(epoch_loss, verbose=True)
    #             # min_loss = min(min_loss, epoch_loss)
    #             # if epoch_loss >= min_loss + patience_thresh:
    #             #     patience_cnt += 1
    #             # else:
    #             #     patience_cnt = 0
    #             #     best_model_wts = copy.deepcopy(model.state_dict())
    #             #     if classify:
    #             #         best_acc = epoch_acc
    #             # if patience_cnt >= patience_before_stopping:
    #             #     stop_bool = True
    #             # # prev_loss = epoch_loss
    #
            if stopping_criterion(stat_dict):
                return model

    #     if stop_bool:
    #         print("Stopping early.")
    #         break
    #
    #     if scheduler is not None:
    #         try:
    #             scheduler.step(epoch_losses['val'][epoch])
    #             print(scheduler.num_bad_epochs)
    #         except AttributeError:
    #             scheduler.step()
    #
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # if classify:
    #     print('Final val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # if classify:
    #     history = dict(losses=epoch_losses, accuracies=epoch_accuracies, final_epoch=epoch)
    # else:
    #     history = dict(losses=epoch_losses, final_epoch=epoch)
    # return model, history
    return model


if __name__ == '__main__':
    ff = models.DenseRandomRegularFF(20, 15, 2, 2, 0.1, 0.4, 'tanh')
    data = tasks.random_classification(20, 20)
    criterion_CEL = nn.CrossEntropyLoss()
    # loss = 0
    # if len(outputs.shape) == 3:
    #     for i1 in range(outputs.shape[1]):
    #         loss += loss_function(outputs[:, i1], labels[:, i1])
    def loss(output, label):
        return criterion_CEL(output, label.long())

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, ff.parameters()), lr=.1)
    train_model(ff, data, loss, optimizer, stopping_epoch=10, out_dir='test')

