import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import typing
    

def train_test_group_split(*arrays, 
                            y,
                            group, 
                            test_size:float=None, 
                            train_size:float=None, 
                            random_state:typing.Union[None, int]=None, 
                            shuffle:bool=True,
                            ):
        '''
        This function returns the train and test data given the
        split and the data. A single ```group``` will not be in
        both the training and testing set. You should use either
        ```test_size``` or ```train_size``` but not both.



        Example
        ---------
        ```
        >>> (X_train, X_test, 
            y_train, y_test, 
            ids_train, ids_test) = train_test_group_split(X, y=y, group=group, test_size=0.33)

        ```


        Arguments
        ---------

        - ```arrays```: ```array-like```, optional:
            The data to split into training and testing sets. The labels and
            the group should be passed to ```y``` and ```group``` respectively.

        - ```y```: ```array-like```, optional:
            Label data with shape ```(n_samples)```, 
            where ```n_samples``` is the number of samples. These are the
            labels that are used to group the data into either the training
            or testing set.

        - ```group```: ```array-like```, optional:
            Event data with shape ```(n_samples)```, 
            where ```n_samples``` is the number of samples. These are the
            group ids that are used to group the data into either the training
            or testing set.
        
        - ```test_size```: ```float```, optional:
            This dictates the size of the outputted test set. This 
            should be used if ```train_size=None```. If no ```test_size```
            or ```train_size``` are given, then ```test_size``` will default
            to ```0.25```
            Defaults to ```None```.

        - ```train_size```: ```float```, optional:
            This dictates the size of the outputted train set. This 
            should be used if ```test_size=None```.
            Defaults to ```None```.

        - ```shuffle```: ```bool```, optional:
            dictates whether the data should be shuffled before the split
            is made.
        
        - ```random_state```: ```None``` or ```int```, optional:
            This dictates the random seed that is used in the random
            operations for this function.



        Returns
        ----------

        - ```split arrays```: ```list```:
            This is a list of the input data, split into the training and
            testing sets. See the Example for an understanding of the 
            order of the outputted arrays.

        
        '''

        # setting defaults for test_size
        assert ~(test_size!=None and train_size!=None), 'Please supply '\
                                                        'either a train_size or a test_size'
        assert len(arrays)>0, 'Please pass arrays to be split.'


        if test_size is None:
            if not train_size is None:
                test_size = 1-train_size
            else:
                test_size = 0.25

        # using the k fold splitter above
        splitter = StratifiedGroupKFold(n_splits=int(1/test_size),
                                        shuffle=shuffle,
                                        random_state=random_state)

        splits = splitter.split(arrays[0], y=y, groups=group)

        # getting the split
        train_idx, test_idx = next(splits)
        
        # creating the output list
        output = []
        for array in arrays:
            output.append(array[train_idx])
            output.append(array[test_idx])
        
        output.append(y[train_idx])
        output.append(y[test_idx])
        output.append(group[train_idx])
        output.append(group[test_idx])
        
        return output