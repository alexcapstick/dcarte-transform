from unittest import result
import numpy as np
import pandas as pd
import functools
import logging


def _split_apply_func(array:np.array, mask:np.array, func):
    '''
    This function makes the call:
    ```func(array[mask], array[~mask])```.
    
    Arguments
    ---------
    
    - ```array```: ```np.array```: 
        The array to split across the first two arguments.
    
    - ```mask```: ```np.array```: 
        The mask that decides which argument part of the array should
        be in. ```True``` dictates that the corresponding element
        should be in the first positional argument in ```func```.
    
    - ```func```: ```function```: 
        The function that is applied to the arguments.
        It should accept arguments like:
        ```func(array1, array2)```.    
    
    
    Returns
    --------
    
    - ```out```: ```Any``` : 
        The function output.
    
    
    '''
    return func(array[mask], array[~mask])





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



# relative median function
def relative_func_delta(array_sample, array_distribution, func):
    funced_sample = func(array_sample)
    funced_distribution = func(array_distribution)
    if funced_distribution == 0:
        return np.nan
    return (funced_sample-funced_distribution)/funced_distribution




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



def datetime_rolling(
                    df:pd.DataFrame, 
                    funcs, 
                    s:str='1d', 
                    w:str='7d', 
                    datetime_col:str='start_date',
                    value_col:str='value', 
                    label:str='left'):
    '''
    This function will roll over a dataframe, with step size
    equal to ```s``` and with a window equal to ```w```, applying
    the functions given to each window.

    This is required as pandas does not allow for a custom step
    size in its ```.rolling()``` function.
    
    
    
    Arguments
    ---------
    
    - ```df```: ```_type_```: 
        The dataframe to apply the rolling function to.
    
    - ```funcs```: ```str```, optional:
        The functions that will be applied to the values.

    - ```s```: ```str```, optional:
        The step size when rolling over the dataframe. 
        Defaults to ```'1d'```.

    - ```w```: ```str```, optional:
        The window size used in the rolling calculation. 
        Defaults to ```'7d'```.
    
    - ```datetime_col```: ```str```, optional:
        The name of the column containing the datetimes.
        This will be passed into ```pandas.to_datetime()``` 
        before operations are applied to it. 
        Defaults to ```'start_date'```.
    
    - ```value_col```: ```str```, optional:
        The column name for the values that will be
        passed to the functions given in ```funcs```. 
        Defaults to ```'value'```.
    
    - ```label```: ```str```, optional:
        The direction used when labelling date time values
        based on each of the steps. This can be in 
        ```['left', 'right']```.
        Defaults to ```'left'```.
    
    
    Returns
    --------
    
    - ```out```: ```_type_``` : 
        A dataframe, with the calculations under
        the column names equal to the functiosn that 
        produce them and the date time of the beginning
        or end of each window, depending on the 
        ```label``` argument.
    
    
    '''
     
    assert label in ['left', 'right'], "Please use label in ['left', 'right']."

    if type(funcs) != list:
        funcs=[funcs]
    
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)

    min_date = df[datetime_col].min()
    max_date =  df[datetime_col].max()
    
    start_date = pd.to_datetime(min_date.date())
    end_date = start_date+pd.Timedelta(w)

    result_dict = {datetime_col: []}
    
    # iterating over the windows
    while end_date<max_date:
        values = df[df[datetime_col].between(start_date, end_date, inclusive='left')][value_col].values
        result_dict[datetime_col].append(end_date if label == 'right' else start_date)
        # iterating over the functions
        for func in funcs:
            if type(func) is functools.partial:
                func_name = func.keywords['func'].__name__
            else:
                func_name = func.__name__

            if not func_name in result_dict:
                result_dict[func_name] = []

            if len(values) == 0:
                result_dict[func_name].append(np.nan)
            else:
                result_dict[func_name].append(func(values))
        start_date += pd.Timedelta(s)
        end_date += pd.Timedelta(s)

    return pd.DataFrame(result_dict)






def datetime_compare_rolling(df:pd.DataFrame, 
                                funcs,
                                s:str='1d', 
                                w_distribution:str='7d', 
                                w_sample:str='1d', 
                                datetime_col:str='start_date', 
                                value_col:str='value', 
                                label:str='left',
                                sorted=False,
                                ):
    '''
    This function will roll over a dataframe, with step size
    equal to ```s```. This function compares the data in 
    ```w_sample``` and ```w_distribution``` by passing the 
    data in them to the functions given in ```funcs```,
    which should have structure:
    ```
    result = func(array_sample, array_distribution)
    ```.
    

    Example
    ---------

    The following would calculate the relative
    change in the median between each day's data 
    and the previous week's data, grouped
    by ID and transition. The calculations
    would also be computed in parallel.

    ```
    from functools import partial
    import dcarte
    from pandarallel import pandarallel as pandarallel_
    from dcarte_transform.utils.progress import tqdm_style, pandarallel_progress

    # loading data
    transitions = dcarte.load('transitions', 'base')
    # filtering out the transitions longer than 5 minutes
    transition_upper_bound = 5*60
    transitions=transitions[transitions['dur']<transition_upper_bound]

    # setting up parallel compute
    pandarallel_progress(desc="Computing transition median deltas", smoothing=0, **tqdm_style)
    pandarallel_.initialize(progress_bar=True)

    # relative median function
    def relative_median_delta(array_sample, array_distribution):
        import numpy as np # required for parallel compute on Windows
        median_sample = np.median(array_sample)
        median_distribution = np.median(array_distribution)
        if median_distribution == 0:
            return np.nan
        return (median_sample-median_distribution)/median_distribution

    # setting up arguments for the rolling calculations
    datetime_compare_rolling_partial = partial(
                                        datetime_compare_rolling, 
                                        funcs=[relative_median_delta], 
                                        s='1d', 
                                        w_distribution='7d', 
                                        w_sample='1d', 
                                        datetime_col='start_date', 
                                        value_col='dur',
                                        label='left',
                                        )

    daily_rel_transitions = (transitions
                            [['patient_id', 'transition', 'start_date', 'dur']]
                            .sort_values('start_date')
                            .dropna()
                            .groupby(by=['patient_id', 'transition',])
                            .parallel_apply(datetime_compare_rolling_partial)
                        )
    daily_rel_transitions['date'] = pd.to_datetime(daily_rel_transitions['start_date']).dt.date
    daily_rel_transitions = (daily_rel_transitions
                                .reset_index(drop=False)
                                .drop(['level_2', 'start_date'], axis=1))

    ```
    
    
    Arguments
    ---------
    
    - ```df```: ```_type_```: 
        The dataframe to apply the rolling function to.
    
    - ```funcs```: ```str```, optional:
        The functions that will be applied to the values.
        This should allow for two arguments to be passed.
        It will be called in the following way:
        ```func(array_distribution, array_sample)```.

    - ```s```: ```str```, optional:
        The step size when rolling over the dataframe. 
        Defaults to ```'1d'```.

    - ```w_distribution```: ```str```, optional:
        The window size for the distribution.
        Defaults to ```'7d'```.
    
    - ```w_sample```: ```str```, optional:
        The window size for the sample.
        Defaults to ```'7d'```.

    - ```datetime_col```: ```str```, optional:
        The name of the column containing the datetimes.
        This will be passed into ```pandas.to_datetime()``` 
        before operations are applied to it. 
        Defaults to ```'start_date'```.
    
    - ```value_col```: ```str```, optional:
        The column name for the values that will be
        passed to the functions given in ```funcs```. 
        Defaults to ```'value'```.
    
    - ```label```: ```str```, optional:
        The direction used when labelling date time values
        based on each of the steps. This can be in 
        ```['left', 'right']```. ```'left'``` will use
        the date from the beginning of the data in ```w_sample```,
        whereas ```'right'``` will use the end datetime of
        the data in ```w_sample```.
        Defaults to ```'left'```.
    
    - ```sorted```: ```bool```, optional:
        If ```False```, this function will sort
        the dataframe on the ```datetime_col``` before
        performing any calculations. If the dataframe is
        already sorted then please give ```sorted=True```.
        Defaults to ```False```.

    
    Returns
    --------
    
    - ```out```: ```_type_``` : 
        A dataframe, with the calculations under
        the column names equal to the functiosn that 
        produce them and the date time of the beginning
        or end of each window, depending on the 
        ```label``` argument.
    
    
    '''

    assert label in ['left', 'right'], "Please use label in ['left', 'right']."
    
    # required for parallel compute on Windows
    import pandas as pd
    from collections import OrderedDict
    import numpy as np
    import functools
    
    # ensuring the data frame is sorted
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    if not sorted:
        df = df.sort_values(datetime_col)

    min_date = df[datetime_col].min()
    max_date =  df[datetime_col].max()
    
    start_date = pd.to_datetime(min_date.date())
    distribution_end_date = start_date+pd.Timedelta(w_distribution)
    end_date = start_date+pd.Timedelta(w_distribution)+pd.Timedelta(w_sample)
    
    result_dict = OrderedDict([])
    result_dict[datetime_col] = []
    
    # iterating over the windows
    while end_date<max_date:
        
        # collating data for both windows
        all_window_values = df[df[datetime_col].between(start_date, end_date, inclusive='left')]
        distribution_values = (all_window_values
                                [all_window_values[datetime_col].between(start_date, 
                                                                            distribution_end_date, 
                                                                            inclusive='left')])[value_col].values
        sample_values = (all_window_values
                            [all_window_values[datetime_col].between(distribution_end_date, 
                                                                        end_date, 
                                                                        inclusive='left')])[value_col].values
        
        result_dict[datetime_col].append(end_date if label == 'right' else distribution_end_date)
        
        # iterating over the windows
        for func in funcs:

            if type(func) is functools.partial:
                func_name = func.keywords['func'].__name__
            else:
                func_name = func.__name__
            
            if not f'{func_name}_relative_delta' in result_dict:
                result_dict[f'{func_name}_relative_delta'] = []


            if len(distribution_values) == 0 or len(sample_values) == 0:
                result_dict[f'{func_name}_relative_delta'].append(np.nan)
            else:
                func_result = func(sample_values, distribution_values)
                result_dict[f'{func_name}_relative_delta'].append(func_result)
        start_date += pd.Timedelta(s)
        end_date += pd.Timedelta(s)
        distribution_end_date += pd.Timedelta(s)

    
    return pd.DataFrame(result_dict)



if __name__=='__main__':
    from functools import partial
    import dcarte
    from pandarallel import pandarallel as pandarallel_
    from dcarte_transform.utils.progress import tqdm_style, pandarallel_progress

    # loading data
    transitions = dcarte.load('transitions', 'base')
    # filtering out the transitions longer than 5 minutes
    transition_upper_bound = 5*60
    transitions=transitions[transitions['dur']<transition_upper_bound]

    # setting up parallel compute
    pandarallel_progress(desc="Computing transition median deltas", smoothing=0, **tqdm_style)
    pandarallel_.initialize(progress_bar=True)

    # relative median function
    def relative_median_delta(array_sample, array_distribution):
        import numpy as np # required for parallel compute on Windows
        median_sample = np.median(array_sample)
        median_distribution = np.median(array_distribution)
        if median_distribution == 0:
            return np.nan
        return (median_sample-median_distribution)/median_distribution

    # setting up arguments for the rolling calculations
    datetime_compare_rolling_partial = partial(
                                        datetime_compare_rolling, 
                                        funcs=[relative_median_delta], 
                                        s='1d', 
                                        w_distribution='7d', 
                                        w_sample='1d', 
                                        datetime_col='start_date', 
                                        value_col='dur',
                                        label='left',
                                        )

    daily_rel_transitions = (transitions
                            [['patient_id', 'transition', 'start_date', 'dur']]
                            .sort_values('start_date')
                            .dropna()
                            .groupby(by=['patient_id', 'transition',])
                            .parallel_apply(datetime_compare_rolling_partial)
                        )
    daily_rel_transitions['date'] = pd.to_datetime(daily_rel_transitions['start_date']).dt.date
    daily_rel_transitions = (daily_rel_transitions
                                .reset_index(drop=False)
                                .drop(['level_2', 'start_date'], axis=1))