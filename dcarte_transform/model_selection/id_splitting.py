'''
Splitting data ID functions.
'''

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import typing

from .base_splitting import train_test_group_split

# This class wraps the StratifiedGroupKFold class from sklearn with some
# information that is useful for our context.
class StratifiedPIDKFold(StratifiedGroupKFold):
    def __init__(self, 
                    n_splits:int=5, 
                    shuffle:bool=False, 
                    random_state:typing.Union[None, int]=None):
        '''
        This function allows you to split the dataset, such that the proportion
        of labels across the training and testing sets are as equal as possible,
        whilst maintaining that no single PID appears in both the training and 
        testing set.


        Example
        ---------
        ```
        >>> splitter = StratifiedPIDKFold()
        >>> splits = splitter.split(X, y.astype(int), ids)
        >>> for train_idx, test_idx in splits:
                X_train, y_train, ids_train = X[train_idx], y[train_idx], ids[train_idx]
                X_test, y_test, ids_test = X[test_idx], y[test_idx], ids[test_idx]

        ```

        
        Arguments
        ---------

        - `n_splits`: `int`, optional:
            This is the number of splits to produce.

        - `shuffle`: `bool`, optional:
            dictates whether the data should be shuffled before the splits
            are made.
        
        - `random_state`: `None` or `int`, optional:
            This dictates the random seed that is used in the random
            operations for this class.
        
        
        '''

        super(StratifiedPIDKFold, self).__init__(n_splits=n_splits, 
                                                    shuffle=shuffle,
                                                    random_state=random_state)

        return
    
    def split(self, X, y, pid):
        '''
        This function builds the splits and returns a generator that can
        be iterated over to produce the training and testing indices.


        Arguments
        ---------

        - `X`: `array-like`, optional:
            Training data with shape `(n_samples,n_features)`, 
            where `n_samples` is the number 
            of samples and `n_features` is the number of features.

        - `y`: `array-like`, optional:
            Label data with shape `(n_samples)`, 
            where `n_samples` is the number of samples. These are 
            the labels that are used to stratify the data. This 
            must be an array of integers.

        - `pid`: `array-like`, optional:
            PID data with shape `(n_samples)`, 
            where `n_samples` is the number of samples. These are the
            ids that are used to group the data into either the training
            or testing set.


        Returns
        ----------

        - `splits`: `generator`:
            This is the generator containing the indices of the splits.

        
        '''


        return super(StratifiedPIDKFold, self).split(X=X, y=y, groups=pid)
        
    


def train_test_pid_split(*arrays, 
                            y,
                            pid, 
                            test_size:float=None, 
                            train_size:float=None, 
                            random_state:typing.Union[None, int]=None, 
                            shuffle:bool=True,
                            ):
    '''
    This function returns the train and test data given the
    split and the data. A single `pid` will not be in
    both the training and testing set. You should use either
    `test_size` or `train_size` but not both.



    Example
    ---------
    ```
    >>> (X_train, X_test, 
        y_train, y_test, 
        ids_train, ids_test) = train_test_pid_split(X, y=y, pid=pid, test_size=0.33)

    ```


    Arguments
    ---------

    - `arrays`: `array-like`, optional:
        The data to split into training and testing sets. The labels and
        the PIDs should be passed to `y` and `pid` respectively.

    - `y`: `array-like`, optional:
        Label data with shape `(n_samples)`, 
        where `n_samples` is the number of samples. These are the
        labels that are used to group the data into either the training
        or testing set.

    - `pid`: `array-like`, optional:
        PID data with shape `(n_samples)`, 
        where `n_samples` is the number of samples. These are the
        ids that are used to group the data into either the training
        or testing set.
    
    - `test_size`: `float`, optional:
        This dictates the size of the outputted test set. This 
        should be used if `train_size=None`. If no `test_size`
        or `train_size` are given, then `test_size` will default
        to `0.25`
        Defaults to `None`.

    - `train_size`: `float`, optional:
        This dictates the size of the outputted train set. This 
        should be used if `test_size=None`.
        Defaults to `None`.

    - `shuffle`: `bool`, optional:
        dictates whether the data should be shuffled before the split
        is made.
    
    - `random_state`: `None` or `int`, optional:
        This dictates the random seed that is used in the random
        operations for this function.



    Returns
    ----------

    - `split arrays`: `list`:
        This is a list of the input data, split into the training and
        testing sets. See the Example for an understanding of the 
        order of the outputted arrays.

    
    '''
    outputs = train_test_group_split(*arrays, 
                                    y=y,
                                    group=pid, 
                                    test_size=test_size, 
                                    train_size=train_size, 
                                    random_state=random_state, 
                                    shuffle=shuffle,
                                    )
    return outputs

