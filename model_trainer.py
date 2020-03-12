from typing import Callable, Union, Dict, Optional
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
# from pdb import set_trace as stp

from . import models
# import models
from . import tasks

# import tasks

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
            self.running_accuracy_epoch = 0.

        num_terms_tot = self.batches_per_epoch*epoch+(batch+1)
        num_terms_epoch = batch+1
        # print(n)
        self.running_avg_loss_tot = (self.running_avg_loss_tot*((num_terms_tot-1)/num_terms_tot)
                                     +stat_dict['loss']/num_terms_tot)
        self.running_avg_loss_epoch = (self.running_avg_loss_epoch*((num_terms_epoch-1)/num_terms_epoch)
                                       +stat_dict['loss']/num_terms_epoch)

        # print(stat_dict['loss'])
        # print(self.running_avg_loss_epoch)
        if self.accuracy:
            out_class = torch.argmax(stat_dict['outputs'], dim=1)
            accuracy = torch.mean((out_class == stat_dict['targets']).double())

            self.running_accuracy_epoch = (self.running_accuracy_epoch*((num_terms_epoch-1)/num_terms_epoch)
                                           +accuracy/num_terms_epoch)

        if stat_dict['epoch_end']:  # We've reached the end of an epoch
            self.epoch_losses.append(self.running_avg_loss_epoch)
            # print()
            print(f"Average {self.phase} loss over this epoch: {self.running_avg_loss_epoch}")
            if self.accuracy:
                print(f"Average {self.phase} accuracy over this epoch: {self.running_accuracy_epoch}")

    def export_stats(self):
        out_stats = dict(epoch_losses=self.epoch_losses)
        return out_stats

class LearningScheduler:
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

        num_terms_epoch = batch+1
        self.running_avg_loss_epoch = (self.running_avg_loss_epoch*((num_terms_epoch-1)/num_terms_epoch)
                                       +stats_dict['loss']/num_terms_epoch)

        if stats_dict['epoch_end'] and phase == 'val':  # We've reached the end of an epoch and are validating
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

def default_return_model_criterion(stat_dict):
    return stat_dict['final_epoch']

def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader],
                loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                optimizer: Optimizer,
                starting_epoch: int = 0,
                stopping_epoch: int = 5,
                out_dir: Optional[str] = None,
                learning_scheduler=None,
                return_model_criterion: Optional[object] = None,
                save_model_criterion: Optional[Callable[[Dict[int, float]], bool]] = None,
                stopping_criterion: Optional[Callable[[Dict[int, float]], bool]] = None,
                stats_trackers: Union[None, Dict[str, Callable], str] = None):
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
        An instantiation of a class that inherets from a torch Optimizer object. Used to train the model. Examples
        are instantiations of torch.optim.SGD and of torch.optim.RMSprop
    starting_epoch : int
        CAUTION: Not implemented. Don't touch.
    stopping_epoch : int
        The maximum number of epochs used to train. An epoch size is defined by the length of the training dataset as
        contained in dataloaders: "len(dataloaders['train'])"
    out_dir : Optional[str, Path]
        The output directory in which to save the model parameters through training.
    stats_trackers : Union[None, Dict[Callable], str]
    learning_scheduler : object
        An instantiation of a torch scheduler object, like those found in torch.optim.lr_scheduler.
    return_model_criterion : Optional[Callable[[Dict], bool]]
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
    Dict
        A dictionary that holds the learning_scheduler and stats_trackers. These are useful for starting a training
        session where a previous one left off.
    Callable
        The learning scheduler object. This is useful for starting a training session where a previous one left off.

    """
    out_dir = Path(out_dir)
    device = torch.device("cpu")
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in dataloaders}
    batches_per_epoch = {x: int(dataset_sizes[x]/dataloaders[x].batch_size) for x in ['train', 'val']}
    checkpoint_ctr = starting_epoch

    model.eval()

    if learning_scheduler is None:
        def learning_scheduler(stat_dict, phase):
            pass

    if save_model_criterion is None:
        save_model_criterion = default_save_model_criterion
    if stopping_criterion is None:
        stopping_criterion = default_stopping_criterion
    if return_model_criterion is None:
        return_model_criterion = default_return_model_criterion

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

    models_to_return = []
    for epoch in range(starting_epoch, stopping_epoch+1):  # epoch 0 corresponds with model before training
        print()
        print('Epoch {}/{}'.format(epoch, stopping_epoch))
        print('-'*10)

        val_batch_num = 0
        for phase in ['train', 'val']:  # First, the train loop, then the validation loop
            training = phase == 'train'
            validating = phase == 'val'
            if training:
                model.train()
            elif validating:
                model.eval()
            with torch.set_grad_enabled(epoch > 0 and training):  # track history only if training and epoch > 0
                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    model.zero_grad()

                    # forward
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)

                    loss_val = loss.item()
                    stat_dict['loss'] = loss_val
                    stat_dict['epoch'] = epoch
                    stat_dict['batch'] = val_batch_num
                    stat_dict['outputs'] = outputs
                    stat_dict['targets'] = targets

                    if dataset_sizes[phase]%batch_sizes[phase] == 0 or dataloaders[phase].drop_last == True:
                        if val_batch_num == batches_per_epoch[phase]-1:
                            stat_dict['epoch_end'] = True
                        else:
                            stat_dict['epoch_end'] = False
                    else:
                        if inputs.shape[0] != batch_sizes[phase]:
                            stat_dict['epoch_end'] = True
                        else:
                            stat_dict['epoch_end'] = False
                    if epoch == stopping_epoch:
                        stat_dict['final_epoch'] = True

                    if training:
                        if epoch > 0:
                            loss.backward()
                            optimizer.step()
                    elif validating:
                        if save_model_criterion(stat_dict) and out_dir is not None:
                            save_checkpoint({'model_state_dict': model.state_dict(),
                                             'optimizer_state_dict': optimizer.state_dict(),
                                             'learning_scheduler_state_dict': learning_scheduler.scheduler.state_dict()},
                                            filename=out_dir/'check_{}'.format(checkpoint_ctr))
                            checkpoint_ctr += 1
                        if return_model_criterion(stat_dict):
                            models_to_return.append(model)
                        val_batch_num = val_batch_num+1
                    if epoch > 0:
                        learning_scheduler(stat_dict, phase)
                    stats_trackers[phase](stat_dict)

            if stopping_criterion(stat_dict):
                return models_to_return, {x: stats_trackers[x].export_stats() for x in ['train', 'val']}

    #
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # if classify:
    #     print('Final val Acc: {:4f}'.format(best_acc))
    export_dict = {x: stats_trackers[x].export_stats() for x in ['train', 'val']}
    training_log_and_machinery = dict(stats_history=export_dict, stats_trackers=stats_trackers,
                                      learning_scheduler=learning_scheduler, optimizer=optimizer)

    return models_to_return, training_log_and_machinery

if __name__ == '__main__':
    # ff = models.DenseRandomRegularFF(20, 15, 2, 2, 0.1, 0.4, 'tanh')
    ff = models.DenseRandomFF(20, 15, 2, 2, 0.1, 0.4, 'tanh')
    data = tasks.classification.random_classification(110, 20)
    criterion_CEL = nn.CrossEntropyLoss()

    # loss = 0
    # if len(outputs.shape) == 3:
    #     for i1 in range(outputs.shape[1]):
    #         loss += loss_function(outputs[:, i1], labels[:, i1])
    def loss(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return criterion_CEL(output, label.long())

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, ff.parameters()), lr=.1)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=5, threshold=1e-7,
                                                          threshold_mode='abs', min_lr=0, verbose=True)
    lr_sched = LearningScheduler(lr_sched)
    trained_models, stats, stats_trackers, learning_scheduler = train_model(ff, data, loss, optimizer,
                                                                            learning_scheduler=lr_sched,
                                                                            stopping_epoch=10, out_dir='test')
    out = train_model(trained_models[0], data, loss, optimizer, starting_epoch=11, stopping_epoch=15, out_dir='test',
                      learning_scheduler=learning_scheduler, stats_trackers=stats_trackers)

    print
