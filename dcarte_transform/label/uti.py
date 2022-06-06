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





def get_labels(days_either_side:int=0) -> pd.DataFrame:
    '''
    This function will return the UTI labels.
    
    
    
    Arguments
    ---------
    
    - ```days_either_side```: ```int```, optional:
        The number of days either side of a label that will be given the same label.
        If these days overlap, the label produced by the most recent true date will
        be used for the overlapping days.
        Defaults to ```0```.
    
    
    
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
        df_labels = df_labels.groupby(['patient_id', 
                                        'date', 
                                        'outcome']).apply(dates_either_side_group_by
                                                ).reset_index(drop=True)

    return df_labels.reset_index(drop=True)







@dcarte.utils.timer('mapping UTI labels')
def label(df:pd.DataFrame,  days_either_side:int=0, return_event=False) -> pd.DataFrame:
    '''
    This function will label the input dataframe based on the uti data 
    in ```procedure```.
    
    
    
    Arguments
    ----------
    
    - ```df```: ```pandas.DataFrame```:
        Unlabelled dataframe, must contain columns ```[patient_id, start_date]```, where ```patient_id``` is the
        ids of participants and ```start_date``` is the time of the sensors.

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
        This is a dataframe containing the original data along with a new column, ```'labels'```,
        which contains the labels. If ```return_event=True```, a column titled ```event``` will be 
        added which contains unique IDs for each of the UTI episodes.
    
    
    
    '''
    assert type(days_either_side) == int, 'days_either_side must be an integer.'

    df_labels = get_labels(days_either_side=days_either_side)

    df['date'] = pd.to_datetime(df['start_date']).dt.date

    # ensure the joining series have the same types
    logging.info('Making sure the pandas columns that are used for the join have the same type.')
    id_type = df['patient_id'].dtype
    date_type = df['date'].dtype
    df_labels['patient_id'] = df_labels['patient_id'].astype(id_type)
    df_labels['date'] = df_labels['date'].astype(date_type)
    
    if return_event:
        df_labels['uti_event'] = [uuid.uuid4() for _ in range(df_labels.shape[0])]

    # performing merge to add labels to the data
    logging.info('Performing merge with data and data labels.')
    df_labelled = pd.merge(df, df_labels, how='left', on=['patient_id', 'date']).drop('date', axis=1).copy()
    df_labelled = df_labelled.rename(columns={'outcome': 'uti_label'})

    return df_labelled







if __name__=='__main__':
    data = dcarte.load('activity', 'raw')
    data_labelled = label(data, days_either_side=2)
    n_positives = np.sum(data_labelled['uti_label']==True)
    n_negatives = np.sum(data_labelled['uti_label']==False)
    print(f'There are {n_positives} positively and {n_negatives} negatively labelled rows.')
