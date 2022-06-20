from tabnanny import verbose
from typing_extensions import OrderedDict
import numpy as np
import torch
import sys
import time
import seaborn as sns
sns.set('talk')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import typing
import tqdm
from collections import OrderedDict
from ..utils.progress import tqdm_style





def loss_plot(loss, val_loss = None, n_epochs = None):
    '''
    For an array of loss values, this function will plot
    a simple loss plot of the results.

    Arguments
    ---------

    - ```loss```: ```list``` or ```np.array```:
        The loss values to plot. Each value
        should represent the loss on that step.
    
    - ```val_loss```: ```list``` or ```np.array```:
        The validation loss values to plot. Each value
        should represent the validation loss on that step.
        If ```None```, no validation loss line will be drawn.
        Defaults to ```None```.

    - ```n_epochs```: ```int```:
        The total number of steps that the model
        will be trained for. This is used to set the 
        bounds on the figure. If ```None```, then
        the limits will be based on the data given.
        Defaults to ```None```.

    Returns
    ---------

    - ```fig```: ```matplotlib.pyplot.figure```:
        The figure containing the axes, which contains
        the plot.
    
    - ```ax```: ```matplotlib.pyplot.axes```:
        The axes containing the plot.


    '''

    # set the plotting area
    fig, ax = plt.subplots(1,1,figsize = (15,5))
    # plot the data
    ax.plot(np.arange(len(loss))+1,loss, label = 'Training Loss')
    if not val_loss is None:
        ax.plot(np.arange(len(val_loss))+1, val_loss, label = 'Validation Loss')
    # label the plots
    ax.set_title('Loss per Sample on Each Step')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Step')
    ax.legend()

    # set limits if the n_epochs is known
    if not n_epochs is None:
        ax.set_xlim(0,n_epochs)

    return fig, ax






















class BasicModelFitter:
    def __init__(self, 
                    model, 
                    device:str='auto', 
                    verbose:bool=False, 
                    model_path:typing.Union[None, str]=None, 
                    result_path:typing.Union[None, str]=None, 
                    model_name:str='', 
                    metrics_track:typing.Union[None, list]=None,
                    writer:torch.utils.tensorboard.SummaryWriter=None,
                    ):
        '''
        This class can be used to fit a model and perform inference.


        Arguments
        ---------
        - ```model```: pytorch model
            This is the pytorch model that can be fit
            and have inference done using.

        - ```device```: ```str``` (optional):
            This is the device name that the model will be trained on. 
            Most common arguments here will be ```'cpu'``` 
            or ```'cuda'```. ```'auto'``` will 
            pick  ```'cuda'``` if available, otherwise
            the training will be performed on ```'cpu'```.
            Defaults to ```'auto'```.
        
        - ```verbose```: ```bool``` (optional):
            Allows the user to specify whether progress
            should be printed as the model is training.
            Defaults to ```False```.

        - ```model_path```: ```str``` or ```None``` (optional):
            Path to the directory in which the models will be saved
            after training. If ```None```, no models are saved.
            If specifying a path, make sure that this path exists.
            Defaults to ```None```.
        
        - ```model_name```: ```str```:
            The name of the model. This is used when saving
            results and the model.
            Defaults to ```''```
        
        - ```metrics_track```: ```list``` of ```str``` or ```None```:
            List of strings containing the names of the 
            metrics to be tracked. Acceptable values are in
            ```['accuracy']```. Loss is tracked by default.
            Defaults to ```None```.
            
            - ```'accuracy'``` reports the mean accuracy over 
                an epoch AFTER the model has been trained on the examples.
                ```'accuracy'``` is accessible via the attributes 
                ```.train_accuracy``` and ```.val_accuracy```.

            Defaults to ```[]```.
        
        - ```writer```: ```torch.utils.tensorboard.SummaryWriter```:
            This is the tensorboard writer that is used to track
            metrics as the model is training. If a writer is not
            passed as an argument then one is assigned with
            the current date and time, and ```model_name``` as its title.

        '''


        self.model = model
        self.train_loss = {}
        self.val_loss = {}
        if type(metrics_track) is str:
            metrics_track = [metrics_track]
        elif metrics_track is None:
            metrics_track = []
        self.track_accuracy = True if 'accuracy' in metrics_track else False
        self.train_accuracy = {}
        self.val_accuracy = {}
        self.n_trains = -1
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.verbose = verbose
        self.model_save = False if model_path is None else True
        self.result_save = False if result_path is None else True
        
        # make sure paths are of the correct format
        if self.model_save:
            if len(model_path) == 0:
                model_path = './'
            elif model_path[-1] != '/':
                model_path += '/'
        if self.result_save:
            if len(result_path) == 0:
                result_path = './'
            elif result_path[-1] != '/':
                result_path += '/'

        self.model_path = model_path
        self.model_name = model_name
        self.result_path = result_path

        # setting tensorboard writer
        if writer is None:
            self.writer = SummaryWriter(comment='-'+model_name)
        else:
            self.writer = writer
        
        self.metrics = {}

        return


    def _fit_traditional_batch(self, data):

        if hasattr(self.model, 'batch_start'):
            self.model.batch_start(self)
        
        self.optimizer.zero_grad()

        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        start_forward_backward_time = time.time()
        # ======= forward ======= 
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        # ======= backward =======
        start_optim_time = time.time()
        loss.backward()
        self.optimizer.step()

        # ======= measuring metrics =======
        end_optim_time = time.time()
        end_forward_backward_time = time.time()

        self.writer.add_scalar('Training Loss', loss, self.step)
        self.writer.add_scalar('Optimiser Time', 
                                end_optim_time-start_optim_time, self.step)
        self.writer.add_scalar('Forward and Backward Time', 
                                end_forward_backward_time-start_forward_backward_time, self.step)

        # assuming the loss function produces mean loss over all instances
        self.epoch_loss += loss.item()*len(inputs)
        self.instances += len(inputs)

        if self.track_accuracy:
            prediction = outputs.argmax(dim=1)
            correct = torch.sum(prediction == labels).item()
            self.epoch_training_correct += correct
            self.writer.add_scalar('Training Accuracy', correct/len(inputs), self.step)
        self.step += 1
        
        if hasattr(self.model, 'batch_end'):
            self.model.batch_end(self)

        self.tqdm_progress.update(1)
        self.tqdm_progress.set_postfix(ordered_dict=self.tqdm_postfix, refresh=True)

        return



    def _validation(self, val_loader):
        val_loss = 0
        with torch.no_grad():
            epoch_validation_correct = 0
            instances = 0
            if hasattr(self.model, 'val_start'):
                self.model.val_start(self)

            for nb, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # perform inference
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()*len(inputs)
                instances += len(inputs)
                
                if self.track_accuracy:
                    prediction = self.model(inputs).argmax(dim=1)
                    correct = torch.sum(prediction == labels).item()
                    epoch_validation_correct += correct

        if hasattr(self.model, 'val_end'):
            self.model.val_end(self)

        epoch_val_loss = val_loss/instances
        self.tqdm_postfix['Val Loss']= f'{epoch_val_loss:.2e}'
        self.writer.add_scalar('Validation Loss', epoch_val_loss, self.step)

        if not 'Epoch Validation Loss' in self.metrics:
            self.metrics['Epoch Validation Loss'] = []
        self.metrics['Epoch Validation Loss'].append(epoch_val_loss)

        if self.track_accuracy: 
            epoch_validation_accuracy = epoch_validation_correct/instances
            self.writer.add_scalar('Validation Accuracy', epoch_validation_accuracy, self.step)
            self.tqdm_postfix['Val Acc'] = f'{epoch_validation_accuracy*100:.1f}'
            if not 'Epoch Validation Accuracy' in self.metrics:
                self.metrics['Epoch Validation Accuracy'] = []
            self.metrics['Epoch Validation Accuracy'].append(epoch_validation_accuracy)


        return 



    def _fit_epoch(self, train_loader, val_loader=None):
        
        if hasattr(self.model, 'epoch_start'):
                self.model.epoch_start(self)

        self.epoch_loss = 0
        self.instances = 0
        if self.track_accuracy:
            self.epoch_training_correct = 0

        # train over batches
        for nb, data in enumerate(train_loader):
            if self.source_fit:
                self._fit_source_traditional_batch(data)
            else:
                self._fit_traditional_batch(data)

        if not self.train_scheduler is None:
            self.train_scheduler.step()

        self.epoch_loss = self.epoch_loss/self.instances
        if not 'Epoch Train Loss' in self.metrics:
            self.metrics['Epoch Train Loss'] = []
        self.metrics['Epoch Train Loss'].append(self.epoch_loss)

        self.tqdm_postfix['Loss']= f'{self.epoch_loss:.2e}'

        if self.track_accuracy: 
            epoch_training_accuracy = self.epoch_training_correct/self.instances
            self.tqdm_postfix['Acc'] = f'{epoch_training_accuracy*100:.1f}'
            if not 'Epoch Train Accuracy' in self.metrics:
                self.metrics['Epoch Train Accuracy'] = []
            self.metrics['Epoch Train Accuracy'].append(epoch_training_accuracy)
        
        # ======= validation =======
        self.model.eval()
        if self.val_too:
            if self.source_fit:
                self._source_validation(val_loader=val_loader)
            else:
                self._validation(val_loader=val_loader)
        self.model.train()



        if hasattr(self.model, 'epoch_end'):
            self.model.epoch_end(self)
        
        # if results saving is true, save the graph.
        if self.result_save:
            if self.val_too:
                fig, ax = loss_plot(self.metrics['Epoch Train Loss'], self.metrics['Epoch Validation Loss'], n_epochs=self.n_epochs)
            else:
                fig, ax = loss_plot(self.metrics['Epoch Train Loss'], n_epochs=self.n_epochs)
            fig.savefig(self.result_path + 'loss_plot-{}.pdf'.format(self.model_name), bbox_inches='tight')
            plt.close()

        return


    def fit(self, 
            train_loader, 
            n_epochs, 
            criterion, 
            optimizer, 
            val_loader=None,
            train_scheduler=None,
            source_fit=False,
            ):
        '''
        This fits the model.

        Arguments
        ---------
            
        - ```train_loader```: ```torch.utils.data.DataLoader```:
            Data loader for the training data. Each iteration 
            should contain the inputs and the targets.

        - ```n_epochs```: ```int```:
            This is the number of epochs to run the training for.
        
        - ```criterion```: pytorch loss function:
            This is the loss function that will be used in the training.
        
        - ```optimizer```: pytorch optimiser:
            This is the optimisation method used in the training.
        
        - ```val_loader```: ```torch.utils.data.DataLoader``` (optional):
            Data loader for the validation data. Each iteration 
            should contain the inputs and the targets.
            If ```None``` then no validation tests will be performed
            as the model is training.
            Defaults to ```None```.
        
        - ```train_scheduler```: ```torch.optim.lr_scheduler``` (optional):
            Learning rate scheduler for training. 
            Defaults to ```None```.
        
        - ```source_fit```: ```bool``` (optional):
            This argument tells the class whether sources are available in 
            the train and validation loaders and passes them to the optimizer
            during training.
            Defaults to ```False```.


        Returns
        ---------

        - ```model```: pytorch model
            This returns the pytorch model after being 
            fitted using the arguments given.


        '''
        # attributes required for training
        self.n_trains += 1
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_scheduler = train_scheduler
        self.n_epochs = n_epochs
        self.val_too = not val_loader is None # true if val_loader is not None
        self.val_loss_temp = []
        self.val_accuracy_temp = []
        self.step = 0
        self.source_fit = source_fit
        if self.source_fit:
            self.source_step_dict = {}
        
        # setting model for training and putting on device
        self.model.train()
        self.model.to(self.device)
        if hasattr(self.model, 'traditional_train_start'):
            self.model.traditional_train_start(self)

        self.tqdm_progress = tqdm.tqdm(
                                total=self.n_epochs*len(train_loader), 
                                desc='Training', 
                                dynamic_ncols=True,
                                disable=not self.verbose,
                                miniters=int(len(train_loader)/10)+1,
                                **tqdm_style,
                                )

        self.tqdm_postfix = OrderedDict([])
        # ======= training =======
        for epoch in range(self.n_epochs):
            self.tqdm_postfix['Epoch'] = epoch+1
            self._fit_epoch(train_loader=train_loader, val_loader=val_loader)

        if hasattr(self.model, 'traditional_train_end'):
            self.model.traditional_train_end(self)


        # saving the model to the model path
        if self.model_save:
            save_name = ('{}-epoch_{}'.format(self.model_name, epoch+1)
                            + '-all_trained' )
            torch.save(self.model.state_dict(), self.model_path + save_name + '-state_dict' + '.pth')

        self.tqdm_progress.close()

        return self.metrics


