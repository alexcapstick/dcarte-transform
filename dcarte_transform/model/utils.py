import torch
import torch.nn as nn

###### making numpy or tensors a torch dataset
class MyData(torch.utils.data.Dataset):
    def __init__(self, *inputs):
        self.inputs = inputs
    def __getitem__(self,index):
        if len(self.inputs) == 1:
            return self.inputs[0][index]
        return [x[index] for x in self.inputs]
    def __len__(self):
        return len(self.inputs[0])


def get_optimizer_from_name(name):
    '''
    Get optimizer from name.
    
    
    
    Arguments
    ---------
    
    - ```name```: ```str```: 
        This can be any of ```'adam'```,
        ```'adadelta'```, ```'sgd'```
    
    
    Raises
    ---------
    
        ```NotImplementedError```: If optimizer name
        is not implemented.
    
    Returns
    --------
    
    - ```optimizer```: ```torch.optim``` optimizer.
    
    
    '''
    if name == 'adam':
        return torch.optim.Adam
    if name =='adadelta':
        return torch.optim.Adadelta
    elif name == 'sgd':
        return torch.optim.SGD
    else:
        raise NotImplementedError('Sorry, {} is not a valid optimizer name.'.format(name))


def get_criterion_from_name(name):
    '''
    Get loss function from name.
    
    
    
    Arguments
    ---------
    
    - ```name```: ```str```: 
        This can be any of ```'celoss'```,
        ```'mseloss'```.
    
    
    Raises
    ---------
    
        ```NotImplementedError```: If loss name
        is not implemented.
    
    Returns
    --------
    
    - ```loss_function```.
    
    
    '''
    if name == 'celoss':
        return nn.CrossEntropyLoss() 
    if name == 'mseloss':
        return nn.MSELoss()
