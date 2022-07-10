import pandas as pd
import numpy as np
import dcarte
import datetime
import logging
import uuid


def map_url_to_flag(urls:pd.Series) -> pd.Series:
    '''
    Maps the URLs to flags using:
    ```
    url_mapping = {
        'http://snomed.info/sct|10828004': True,
        'http://snomed.info/sct|260385009': False,
        'http://snomed.info/sct|82334004': None,
    }
    ```
    
    
    Arguments
    ---------
    
    - ```urls```: ```pd.Series```: 
        The urls to map. This is designed to accept 
        a pandas ```Series``` as input.
    
    
    
    Returns
    --------
    
    - ```out```: ```pd.Series``` : 
        The value that the url maps to.
    
    
    '''
    url_mapping = {
        'http://snomed.info/sct|10828004': True,
        'http://snomed.info/sct|260385009': False,
        'http://snomed.info/sct|82334004': None,
    }

    return list(map(url_mapping.get, urls))






def get_labels(days_either_side:int=0, return_event:bool=False) -> pd.DataFrame:
    '''
    This function will return the UTI labels.
    If a single day for a paticular ID contains two different
    labels (usually caused by using ```days_either_side```),
    then both labels are removed.
    
    
    
    Arguments
    ---------
    
    - ```days_either_side```: ```int```, optional:
        The number of days either side of a label that will be given the same label.
        If these days overlap, if the label is the same then the first will be kept.
        If they are different, then neither will be kept.
        Defaults to ```0```.
    
    - ```return_event```: ```bool```, optional:
        This dictates whether another column should be added, with a unique id given to each of the separate
        UTI events. This allows the user to group the outputted data based on events.
        Defaults to ```False```.
    
    
    Returns
    --------
    
    - ```out```: ```pd.DataFrame``` : 
        A dataframe containing the uti labels, with the corresponding patient_id and 
        date.

    '''
    logging.info('Getting uti labels from the procedure file.')
    df_labels = dcarte.load('procedure', 'raw')

    df_labels['notes'] = df_labels['notes'].str.lower()
    df_labels['type'] = df_labels['type'].str.lower()

    logging.info('Filtering the procedure file to include only the uti labels.')

    # filtering to include rows that are about UTIs
    def uti_filter(x):
        keep = False
        type = str(x.type)
        notes = str(x.notes)
        if ('urine' in type
            or 'urinalysis' in type
            or 'uti' in type):
            keep = True
            return keep
        elif ('urine' in notes 
            or 'urinalysis' in notes 
            or 'uti' in notes ):
            keep = True
            return keep
        return keep

    # cleaning data to include uti labels only
    df_labels['keep'] = df_labels[['type', 'notes']].apply(uti_filter, axis=1)
    df_labels = df_labels[df_labels['keep']==True]
    df_labels = df_labels[['patient_id', 'start_date', 'outcome', 'notes']]
    df_labels.columns = ['patient_id', 'date', 'outcome', 'notes']
    
    logging.info('Mapping outcome url to bool or None.')
    # labelling UTIs from url
    df_labels['outcome'] = map_url_to_flag(df_labels['outcome'])

    logging.info('Making uti labels in which'\
                    'the outcome=True but the notes contain the word "negative" False.')
    # accounting for notes contradicting the label
    df_labels[(df_labels['outcome'] == True)
        & (df_labels.notes.str.contains('negative'))]['outcome'] = False
    
    # calculating date and dropping duplicates
    df_labels = df_labels[['patient_id', 'date', 'outcome']]
    df_labels['date'] = pd.to_datetime(df_labels['date']).dt.date
    df_labels = df_labels.dropna()
    df_labels = df_labels.drop_duplicates()
    if return_event:
        df_labels['uti_event'] = [uuid.uuid4() for _ in range(df_labels.shape[0])]

    # extending the label either side of the date
    if not days_either_side == 0:
        logging.info(f'Extending labels using days_either_side={days_either_side}.')
        def dates_either_side_group_by(x):
            date = x['date'].values[0]
            x = [x] * (2 * days_either_side + 1)
            new_date_values = np.arange(-days_either_side, days_either_side + 1)
            new_dates = [date + datetime.timedelta(int(value)) for value in new_date_values]
            x = pd.concat(x)
            x['date'] = new_dates
            return x
        groupby_names = (['patient_id', 'date',  'uti_event', 'outcome'] if return_event
                            else ['patient_id', 'date', 'outcome']) 
        df_labels = df_labels.groupby(groupby_names).apply(dates_either_side_group_by
                                                ).reset_index(drop=True)
    else:
        df_labels.reset_index(drop=True)
    
    # removing the rows with contradictory labels
    df_labels = (df_labels
                    # keeping one of the rows where the labels are the same
                    .drop_duplicates(subset=['patient_id', 'date', 'outcome'], keep='first')
                    # removing rows where the labels are the different
                    .drop_duplicates(subset=['patient_id', 'date'], keep=False))

    return df_labels.reset_index(drop=True)







@dcarte.utils.timer('mapping UTI labels')
def label(df:pd.DataFrame, id_col:str='patient_id', datetime_col:str='start_date', 
            days_either_side:int=0, return_event:bool=False) -> pd.DataFrame:
    '''
    This function will label the input dataframe based on the uti data 
    in ```procedure```.
    
    
    
    Arguments
    ----------
    
    - ```df```: ```pandas.DataFrame```:
        Unlabelled dataframe, must contain columns ```[id_col, datetime_col]```, where ```id_col``` is the
        ids of participants and ```datetime_col``` is the time of the sensors.

    - ```id_col```: ```str```, optional:
        The column name that contains the ID information.
        Defaults to ```'patient_id'```.

    - ```datetime_col```: ```str```, optional:
        The column name that contains the date time information.
        Defaults to ```'start_date'```.

    - ```days_either_side```: ```int```, optional:
        The number of days either side of a label that will be given the same label.
        Defaults to ```0```.
    
    - ```return_event```: ```bool```, optional:
        This dictates whether another column should be added, with a unique id given to each of the separate
        UTI events. This allows the user to group the outputted data based on events.
        Defaults to ```False```.
    
    
    
    Returns
    ---------
    
    - df_labelled: ```pandas.DataFrame```:
        This is a dataframe containing the original data along with a new column, ```'uti_labels'```,
        which contains the labels. If ```return_event=True```, a column titled ```'uti_event'``` will be 
        added which contains unique IDs for each of the UTI episodes.
    
    
    
    '''
    assert type(days_either_side) == int, 'days_either_side must be an integer.'

    df = df.copy()

    df_labels = get_labels(days_either_side=days_either_side, return_event=return_event)

    df['__date__'] = pd.to_datetime(df[datetime_col]).dt.date

    # ensure the joining series have the same types
    logging.info('Making sure the pandas columns that are used for the join have the same type.')
    id_type = df[id_col].dtype
    date_type = df['__date__'].dtype
    df_labels['patient_id'] = (df_labels
                                ['patient_id']
                                .astype(id_type)
                                )
    df_labels = df_labels.rename({'patient_id': id_col}, axis=1)
    df_labels['date'] = df_labels['date'].astype(date_type)
    df_labels = df_labels.rename({'date': '__date__'}, axis=1)

    # performing merge to add labels to the data
    logging.info('Performing merge with data and data labels.')
    df_labelled = pd.merge(df, df_labels, how='left', on=[id_col, '__date__']).drop('__date__', axis=1).copy()
    df_labelled = df_labelled.rename(columns={'outcome': 'uti_label'})

    return df_labelled




def label_number_previous(
    df:pd.DataFrame, 
    id_col:str='patient_id', 
    datetime_col:str='start_date',
    day_delay=1,
    ):
    '''
    This function allows you to label the number of uti positives to date
    for the corresponding ID and date.
    

    Arguments
    ---------

    - ```df```: ```pandas.DataFrame```:
        The dataframe to append the number of previous uti positives to.
    
    - ```id_col```: ```str```, optional:
        The column name that contains the ID information.
        Defaults to ```'patient_id'```.

    - ```datetime_col```: ```str```, optional:
        The column name that contains the date time information.
        Defaults to ```'start_date'```.

    - ```day_delay```: ```str```, optional:
        The number of days after a UTI is detected when the data reflects
        that the ID has had another previous UTI. This is used to ensure
        that the predictive model does not simply learn that to look for 
        when this feature increases.
        Defaults to ```1```.

    
    Returns
    ---------
    
    - df_out: ```pandas.DataFrame```:
        This is a dataframe containing the original data along with a new column, ```'uti_previous'```,
        which contains the number of previous UTIs to date for that ID.

    
    

    '''

    cumulative_pos_labels = (get_labels()
        .query('outcome == True')
        .groupby(['patient_id', 'date'])
        .sum()
        .groupby(level=0)
        .cumsum()
        .reset_index()
        .sort_values(by='date')
        .rename({'outcome': 'previous_uti',
            'date': '__date__'}, axis=1)
        )
    
    cumulative_pos_labels['__date__'] = pd.to_datetime(
        cumulative_pos_labels['__date__']
        )
    
    # this ensures that the number of previous UTIs increases the day_delay
    # after the UTI
    df['__date__'] = pd.to_datetime(df[datetime_col]) - pd.Timedelta(f'{day_delay}day')
    
    df = pd.merge_asof(
        df.sort_values('__date__'),
        cumulative_pos_labels,
        on=['__date__'],
        direction='backward',
        left_by=id_col,
        right_by='patient_id',
        )
    
    df = df.drop('__date__', axis=1)
    df['previous_uti'] = df['previous_uti'].fillna(0)

    return df









if __name__=='__main__':
    data = dcarte.load('activity', 'raw')
    data_labelled = label(data, days_either_side=2)
    n_positives = np.sum(data_labelled['uti_label']==True)
    n_negatives = np.sum(data_labelled['uti_label']==False)
    print(f'There are {n_positives} positively and {n_negatives} negatively labelled rows.')
