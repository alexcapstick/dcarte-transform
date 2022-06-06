import pandas as pd
import numpy as np
import typing
from pydtmc import MarkovChain
import tqdm
import dcarte

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


def build_p_matrix(sequence, return_events=False):
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


def entropy_rate_from_sequence(sequence):
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

    p_matrix = build_p_matrix(sequence)

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
def get_entropy_rate(df: pd.DataFrame, 
                        datetime_col:str='start_date',
                        location_col:str='location_name',
                        sensors:typing.Union[typing.List[str], str] = 'all', 
                        freq:typing.Union[typing.List[str], str]=['day', 'week']) -> pd.DataFrame:
    '''
    This function allows the user to return a pandas.DataFrame with the entropy rate calculated
    for every week. The dataframe must contain ```'patient_id'```, and columns containing the 
    visited location names and the date and time of these location visits.
    
    
    
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
        This is a data frame, in which the entropy rate is located in the ```'value'``` column.
    
    
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

    # daily entropy calculations
    tqdm.tqdm.pandas(desc="Calculating daily entropy")
    if 'day' in freq:
        daily_entropy = df.groupby(by=['patient_id',
                            pd.Grouper(key=datetime_col, freq='1d')])[location_col]\
                                .progress_apply(lambda x: entropy_rate_from_sequence(x.values)).reset_index()
        daily_entropy.columns = ['patient_id', 'date', 'daily_entropy']
        df['date'] = pd.to_datetime(df[datetime_col]).dt.date.astype(np.datetime64)
        df = pd.merge(df, daily_entropy, how='left', on=['patient_id', 'date']).drop('date', axis=1)

    # weekly entropy calculations
    tqdm.tqdm.pandas(desc="Calculating weekly entropy")
    if 'week' in freq:
        df['week'] = compute_week_number(df[datetime_col])
        weekly_entropy = df.groupby(by=['patient_id','week'])[location_col]\
                            .progress_apply(lambda x: entropy_rate_from_sequence(x.values)).reset_index()
        weekly_entropy.columns = ['patient_id', 'week', 'weekly_entropy']        
        df = pd.merge(df, weekly_entropy, how='left', on=['patient_id', 'week']).drop('week', axis=1)

    return df











if __name__ == '__main__':
    data = dcarte.load('activity', 'raw')
    df = get_entropy_rate(data, freq=['day', 'week'])
