from unittest.mock import NonCallableMagicMock
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing
import seaborn as sns
sns.set('talk')
from copy import deepcopy
from .base_model import BaseModel









class CodeLayer(BaseModel):
    '''

    This class can be set up as either an encoder or a decoder section of an autoencoder
    semi-supervised model. Simply supply the arguments to either reduce or increase the 
    size of the final dimension of the input.

    '''
    def __init__(self, n_input, n_output, n_layers = 2, dropout=0.2):
        '''
        Arguments
        ---------
            
            n_input: int
                This is the size of the last dimensions of the input and is the axis along which
                the input will be decoded.

            n_output: int
                This is the size of the last dimension of the output.

            n_layers: int
                The number of layers to get from input to output dimensions.

        '''

        super(CodeLayer, self).__init__()
        
        in_out_list = np.linspace(n_input, n_output, n_layers + 1, dtype = int)

        in_list = in_out_list[:-1][:-1]
        out_list = in_out_list[:-1][1:]

        self.layers = nn.ModuleList([nn.Sequential(
                                            nn.Linear(in_value, out_value), 
                                            nn.Dropout(dropout),
                                            nn.BatchNorm1d(out_value), 
                                            nn.ReLU())
                                     for in_value, out_value in zip(in_list, out_list)])
        
        self.last_layer = nn.Linear(in_out_list[-2], in_out_list[-1])

        return
    
    def forward(self, X):
        '''
        Returns
        ---------
            
            out: tensor
                This is the decoded version of the input.
        '''
        
        out = X
        for layer in self.layers:
            out = layer(out)
        out = self.last_layer(out)

        return out
    









class AEModel(BaseModel):
    '''
    A simple Auto-Encoder model that learns embeddings.
    '''
    def __init__(self,
                    n_input:int,
                    n_embedding:int, 
                    n_layers:int=2, 
                    dropout:float=0.2,
                    optimizer:dict={'adam':{'lr':0.01}},
                    criterion:typing.Union[str,nn.Module]='mseloss',
                    n_epochs=10,
                    **kwargs,
                    ):
        '''
        An auto-encoder model, built to be run similar to sklearn models.

        Example
        ---------
        ```
        ae_model = AEModel(n_input=100, 
                            n_embedding=5, 
                            n_layers=2,
                            n_epochs = 2,
                            verbose=True,
                            batch_size=10,
                            optimizer={'adam':{'lr':0.01}},
                            criterion='mseloss',
                            )

        X = torch.tensor(np.random.random((10000,100))).float()
        X_val = torch.tensor(np.random.random((10000,100))).float()

        training_metrics = ae_model.fit(X=X, X_val=X_val)
        output = ae_model.transform(X_test=X)


        ```


        Arguments
        ---------

        - ```n_input```: ```int```:
            The size of the input feature dimension.

        - ```n_embedding```: ```int```:
            The number of features that the embedding will have.

        - ```n_layers```: ```int```, optional:
            The number of layers in the encoder model. The decoder
            model will have the same number of layers.
            Defaults to ```2```.

        - ```dropout```: ```float```, optional:
            The dropout value in each of the layers.
            Defaults to ```0.2```
        
        - ```optimizer```: ```dict```, optional:
            A dictionary containing the optimizer name as keys and
            a dictionary as values containing the arguments as keys. 
            For example: ```{'adam':{'lr':0.01}}```.
            Defaults to ```{'adam':{'lr':0.01}}```.
        
        - ```criterion```: ```str``` or ```torch.nn.Module```, optional:
            The criterion to use, which can be a string or a function.
            Defaults to ```mseloss```.
        
        - ```n_epochs```: ```int```, optional:
            The number of epochs to run the training for.
            Defaults to ```10```.

        - ```kwargs```: optional:
            These keyword arguments will be passed to 
            ```dcarte_transform.model.base_model.BaseModel```.


        '''

        if 'model_name' in kwargs:
            if kwargs['model_name'] is None:
                self.model_name = f'AE-{n_input}-{n_embedding}-{n_layers}-{dropout}'

        super(AEModel, self).__init__(optimizer=optimizer,
                                        criterion=criterion,
                                        n_epochs=n_epochs,
                                        **kwargs)
        
        
        self.n_input = n_input
        self.n_embedding = n_embedding
        self.n_layers = n_layers
        self.dropout = dropout

        return


    def _build_model(self):
        self.e = CodeLayer(n_input=self.n_input, n_output=self.n_embedding, n_layers=self.n_layers, dropout=self.dropout)
        self.d = CodeLayer(n_input=self.n_embedding, n_output=self.n_input, n_layers=self.n_layers, dropout=self.dropout)
        self.encoding = False
        return

    def forward(self,X):
        out = self.e(X)
        if self.encoding: return out
        out = self.d(out)
        return out


    def encode(self):
        self.train(mode=False)
        self.encoding = True
    
    def decode(self):
        self.encoding = False

    def train(self, mode=True):
        super(AEModel, self).train(mode=mode)

    def eval(self):
        self.train(mode=False)

    def fit(self,
            X:np.array=None, 
            train_loader:torch.utils.data.DataLoader=None,
            X_val:typing.Union[np.array, None]=None,
            val_loader:torch.utils.data.DataLoader=None,
            **kwargs,
            ):
        '''
        This is used to fit the model. Please either use 
        the ```train_loader``` or ```X``` and ```y```.
        This corresponds to using either a torch DataLoader
        or a numpy array as the training data. If using 
        the ```train_loader```, ensure each iteration returns
        ```[X, X]```.

        Arguments
        ---------

        - ```X```: ```numpy.array``` or ```None```, optional:
            The input array to fit the model on.
            Defaults to ```None```.

        - ```y```: ```numpy.array``` or ```None```, optional:
            The target array to fit the model on.
            Defaults to ```None```.

        - ```train_loader```: ```torch.utils.data.DataLoader``` or ```None```, optional:
            The training data, which contains the input and the targets.
            Defaults to ```None```.

        - ```X_val```: ```numpy.array``` or ```None```, optional:
            The validation input to calculate validation 
            loss on when training the model.
            Defaults to ```None```

        - ```X_val```: ```numpy.array``` or ```None```, optional:
            The validation target to calculate validation 
            loss on when training the model.
            Defaults to ```None```

        - ```val_loader```: ```torch.utils.data.DataLoader``` or ```None```, optional:
            The validation data, which contains the input and the targets.
            Defaults to ```None```.

        '''
        
        self._build_model()

        if train_loader is None:
            y = deepcopy(X)
        else:
            y = None
        
        if val_loader is None:
            y_val = deepcopy(X_val)
        else:
            y_val = None

        return super(AEModel, self).fit(train_loader=train_loader,
                                            X=X, 
                                            y=y,
                                            val_loader=val_loader,
                                            X_val=X_val,
                                            y_val=y_val,
                                            **kwargs,
                                            )

    @torch.no_grad()
    def transform(self,
                    X_test:np.array=None, 
                    test_loader:torch.utils.data.DataLoader=None,
                  ):
        '''
        Method for transforming data based on the fit AE.
        
        Arguments
        ---------
        
        - ```X_test```: ```numpy.array``` or ```None```, optional:
            The input array to test the model on.
            Defaults to ```None```.
        
        - ```test_loader```: ```torch.utils.data.DataLoader``` or ```None```, optional: 
            A data loader containing the test data.
            Defaults to ```None```.
        
        
        Returns
        --------
        
        - ```output```: ```torch.tensor``` : 
            The resutls from the predictions
        
        
        '''
        

        self.encode()

        return super(AEModel, self).predict(
                    test_loader=test_loader,
                    X_test=X_test, 
                    )


