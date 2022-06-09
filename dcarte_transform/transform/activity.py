import pandas as pd
import numpy as np
import typing
from pydtmc import MarkovChain
import tqdm
import dcarte
from utils import compute_delta

def compute_week_number(df:pd.DataFrame):
    '''
    Compute the week number from the date.

    Arguments
    ---------

    - ```df```: ```pd.DataFrame```:
        A data frame containing the dates to convert to week numbers.
    
    '''
    df = pd.to_datetime(df, utc=True, infer_datetime_format=True)
    return df.dt.isocalendar().week + (df.dt.isocalendar().year - 2000) * 100


def compute_p_matrix(sequence, return_events=False):
    '''
    This function allows the user to create a stochastic matrix from a 
    sequence of events.
    
    
    Arguments
    ---------
    
    - ```sequence```: ```numpy.array```: 
        A sequence of events that will be used to calculate the stochastic matrix.

    - ```return_events```: ```bool```, optional:
        Dictates whether a list of the events should be returned, in the 
        order of their appearance in the stochastic matrix, ```p_martix```.
        Defaults to ```False```
    
    
    Returns
    --------
    
    - ```p_matrix```: ```numpy.array```: 
        A stochastic matrix, in which all of the rows sum to 1.

    - ```unique_locations```: ```list```:
        A list of the events in the order of their appearance in the stochastic
        matrix, ```p_martix```. This is only returned if ```return_events=True```
    
    
    '''

    # calculating transitions
    sequence_df = pd.DataFrame()
    sequence_df['from'] = sequence[:-1]
    sequence_df['to'] = sequence[1:]
    sequence_df['count'] = 1
    pm = sequence_df.groupby(by=['from','to']).count().reset_index()
    pm_total = pm.groupby(by='from')['count'].sum().to_dict()
    pm['total'] = pm['from'].map(pm_total)
    
    if pm.shape[0] < 2:
        return np.nan
    
    # calculating transition probabilities
    def calc_prob(x):
        return x['count']/x['total']

    pm['probability'] = pm.apply(calc_prob, axis = 1)
    unique_locations = list(np.unique(pm[['from', 'to']].values.ravel()))
    p_matrix = np.zeros((len(unique_locations),len(unique_locations)))

    # calculating p matrix
    for (from_loc, to_loc, probability_loc) in pm[['from', 'to', 'probability']].values:
        i = unique_locations.index(from_loc)
        j = unique_locations.index(to_loc)
        p_matrix[i,j] = probability_loc

    if return_events:
        return p_matrix, unique_locations
    else:
        return p_matrix


def compute_entropy_rate_from_sequence(sequence):
    '''
    This function allows the user to calculate the entropy rate based on
    a sequence of events.



    Arguments
    ---------

    - ```sequence```: ```numpy.array```:
        A sequence of events to calculate the entropy rate on.



    Returns
    --------

    - ```out```: ```float```:
        Entropy rate


    '''

    p_matrix = compute_p_matrix(sequence)

    if type(p_matrix) != np.ndarray:
        return np.nan

    # we do not want to calculate the entropy for those graphs that
    # have a zero in the rows or only have a one in the rows,
    # since this is a consequence of cutting the sequences by a time period
    incomplete_rows = np.diag(p_matrix) == 1
    zero_rows = np.sum(p_matrix,axis=1) == 0
    if any(incomplete_rows) or any(zero_rows):
        return np.nan
    
    mc = MarkovChain(p_matrix)
    return mc.entropy_rate_normalized



@dcarte.utils.timer('calculating entropy')
def compute_entropy_rate(df: pd.DataFrame, 
                        datetime_col:str='start_date',
                        location_col:str='location_name',
                        sensors:typing.Union[typing.List[str], str] = 'all', 
                        freq:typing.Union[typing.List[str], str]=['day', 'week']) -> typing.Union[pd.DataFrame, 
                                                                                        typing.List[pd.DataFrame]]:
    '''
    This function allows the user to return a pandas.DataFrame with the entropy rate calculated
    for every week or day. The dataframe must contain ```'patient_id'```, and columns containing the 
    visited location names and the date and time of these location visits.
    
    
    Example
    ---------

    Note that the daily entropy will always be returned first in the list if two 
    frequencies are given.
    
    ```
    >>> data = dcarte.load('activity','raw')
    >>> daily_entropy, weekly_entropy = compute_entropy_rate(data, freq=['day','week'])
    >>> daily_entropy, weekly_entropy = compute_entropy_rate(data, freq=['week','day'])
    ```


    
    Arguments
    ---------
    
    - ```df```: ```pandas.DataFrame```: 
        A data frame containing ```'patient_id'```, and columns containing the 
        visited location names and the date and time of these location visits.
    
    - ```datetime_col```: ```str```, optional:
        The name of the column that contains the date time of location visits.
        Defaults to ```'start_date'```.

    - ```location_col```: ```str```, optional:
        The name of the column that contains the location names visited.
        Defaults to ```'location_name'```.
    
    - ```sensors```: ```list``` of ```str``` or ```str```: 
        The values of the ```'location'``` column of ```df``` that will be 
        used in the entropy calculations.
        Defaults to ```'all'```.
    
    - ```freq```: ```list``` of ```str``` or ```str```:
        The period to calculate the entropy for. This can either be ```'day'```
        or ```'week'``` or a list containing both.
        Defaults to ```['day', 'week']```
    
    
    
    Returns
    --------
    
    - ```out```: ```pd.DataFrame```: 
        returns a list of data frames containing the weekly and daily entropy
        or  single dataframe if only one ```freq``` was given.
    
    
    '''

    assert len(sensors) >= 2, 'need at least two sensors to calculate the entropy'

    df = df.sort_values(datetime_col).copy()

    if type(freq) == str:
        freq = [freq]

    # filter the sensors
    if isinstance(sensors, list):
        df = df[df.location.isin(sensors)]
    elif isinstance(sensors, str):
        assert sensors == 'all', "Only accept 'all' as a string input for sensors"

    outputs = []

    # daily entropy calculations
    tqdm.tqdm.pandas(desc="Calculating daily entropy")
    if 'day' in freq:
        daily_entropy = df.groupby(by=['patient_id',
                            pd.Grouper(key=datetime_col, freq='1d')])[location_col]\
                                .progress_apply(lambda x: compute_entropy_rate_from_sequence(x.values)).reset_index()
        daily_entropy.columns = ['patient_id', 'date', 'daily_entropy']
        outputs.append(daily_entropy)

    # weekly entropy calculations
    tqdm.tqdm.pandas(desc="Calculating weekly entropy")
    if 'week' in freq:
        weekly_entropy = df.groupby(by=['patient_id',
                            pd.Grouper(key=datetime_col, freq='W-SUN')])[location_col]\
                            .progress_apply(lambda x: compute_entropy_rate_from_sequence(x.values)).reset_index()
        weekly_entropy.columns = ['patient_id', 'date', 'weekly_entropy']
        outputs.append(weekly_entropy)
    
    if len(outputs) > 1:
        return outputs
    else:
        return outputs[0]






@dcarte.utils.timer('calculating daily location frequency')
def compute_daily_location_freq(df:pd.DataFrame, 
                            location:str, 
                            location_col:str='location_name', 
                            datetime_col:str='start_date', 
                            time_range:typing.Union[None, typing.List[str]]=None, 
                            name:typing.Union[None, str]=None)->pd.DataFrame:
    '''
    This function allows you to calculate the frequency of visits to 
    a given location during a given time range, aggregated daily.



    Example
    ---------
    
    To get the frequency of activity in the ```'bathroom1'``` between 
    the times of 00:00 to 08:00 and 20:00 to 00:00 each day, you could
    run the following:
    
    ```
    >>> compute_daily_location_freq(data, 'bathroom1', time_range=['20:00','08:00'])

    ```


    
    Arguments
    ---------

    - ```df```: ```pandas.DataFrame```:
        The data frame containing the location visits to calculate the 
        frequency from.
    
    - ```location```: ```str```:
        The location name to calculate the frequencies for.
    
    - ```location_col```: ```str```, optional:
        The name of the location column that contains the visited location.
        Defaults to ```'location_name'```.
    
    - datetime_col: ```str```, optional:
        The name of the location column that contains the date times
        of the location visits. This will be converted using 
        ```pandas.to_datetime```.
        Defaults to ```'start_date'```.
    
    - time_range: ```None``` or ```list``` of ```str```, optional:
        A time range given here, would allow you filter the frequencies
        by a given time. This allows you to calculate the frequencies
        of visits to a location during the night, for example. Acceptable
        arguments here are ```['[mm]:[ss]','[mm]:[ss]']```, in which 
        the first element of the list is the start time and the second
        element is the end time.
        Defaults to ```None```.
    
    - name: ```str``` or None:
        This argument allows you to name the outputted column that contains
        the frequencies.
        Defaults to ```None```
    
    
    Returns
    ---------

    ```table_of_frequencies```: ```pandas.DataFrame```:
        The table containing the frequencies, with column names
        ```'patient_id'```, ```'date'``` and ```[name]``` or 
        ```[location]_freq```.


    
    '''

    data = df.copy()
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    data = data[data[location_col] == location][['patient_id', datetime_col, location_col]]
    if time_range is None:
        data = data.set_index(datetime_col).reset_index()
    else:
        data = data.set_index(datetime_col).between_time(*time_range, inclusive='left').reset_index()
    data['date'] = data[datetime_col].dt.date
    data = data.groupby(['patient_id', 'date'])[location_col].count().reset_index()
    name = name if name is not None else f'{location}_freq'
    data.columns = ['patient_id', 'date', name]

    return data














if __name__ == '__main__':
    data = dcarte.load('activity', 'raw')
    #entropy_daily, entropy_weekly = compute_entropy_rate(data, freq=['day','week'])
    bathroom_feq = compute_daily_location_freq(data, location='bathroom1')
    bathroom_freq_daytime = compute_daily_location_freq(data, location='bathroom1', time_range=['08:00', '20:00'])
    bathroom_freq_nighttime = compute_daily_location_freq(data, location='bathroom1', time_range=['20:00', '08:00'])
    bathroom_freq_nighttime['bathroom1_freq_ma'] = bathroom_freq_nighttime[['bathroom1_freq']].rolling(3).mean()
    bathroom_freq_nighttime['bathroom1_freq_ma_delta'] = compute_delta(bathroom_freq_nighttime['bathroom1_freq_ma'].values, pad=True)