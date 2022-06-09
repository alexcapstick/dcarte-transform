import numpy as np
import pandas as pd




def moving_average(array:np.array, w:int=3, pad:bool=False,):
    '''
    Calculate the moving average of a 1D array.

    Arguments
    ---------

    - ```array```: ```numpy.array```:
        This is the array to calculate the moving average of.
    
    - ```w```: ```int```, optional:
        This is the window size to use when calculating the
        moving average of the array.
    
    - ```pad```: ```bool```, optional:
        Dictates whether NAN values should be added to the beginning
        of the array, so that the output is of the same shape as 
        ```array```.

    
    Returns
    ---------

    - ```moving_average```: ```numpy.array```:
        An array containing the moving average.
    
    
    
    
    '''
    
    # moving average
    ma = np.convolve(array, np.ones(w), 'valid')/w
    
    # fill in the cut elements with nan
    if pad:
        output = np.empty_like(array, dtype=object)
        output[:] = np.nan
        output[w-1:] = ma
        return output
    
    return ma





def compute_delta(array:np.array, pad:bool=False,):
    '''
    This function allows the user to calculate the proportional change
    between each element in ```x``` and its previous element. This is done
    using the formula:
    ```
    (x_{i} - x_{i-1})/x_{i-1}
    ```
    
    Arguments
    ---------

    - ```x```: ```numpy.array```:
        The array to calculate the delta values on.
    
    - ```pad```: ```bool```, optional:
        Dictates whether NAN values should be added to the beginning
        of the array, so that the output is of the same shape as 
        ```array```.

    
    Returns
    ---------
    
    - ```delta_values```: ```pandas.Series```:
        An array containing the delta values.

    
    '''

    delta = (array[1:]-array[:-1])/array[:-1]

    # fill in the cut elements with nan
    if pad:
        output = np.empty_like(array, dtype=object)
        output[:] = np.nan
        output[1:] = delta
        return output


    return delta