import pandas as pd
import numpy as np
import dcarte
import datetime
import logging
import uuid
import typing



def _label(
    df:pd.DataFrame, 
    label_function:typing.Callable, 
    id_col:str='patient_id', 
    datetime_col:str='start_date',
    ) -> pd.DataFrame:
    '''
    This function will label the input dataframe based on the given :code:`label_name` 
    in :code:`procedure`.

    Arguments
    ----------
    
    - df:  pandas.DataFrame: 
        Unlabelled dataframe, must contain columns :code:`[id_col, datetime_col]`, where :code:`id_col` is the
        ids of participants and :code:`datetime_col` is the time of the sensors.

    - label_name:  str: 
        The name of the label to add to the column.

    - label_function:  typing.Callable: 
        The labelling function to use. This should return the labels in a dataframe
        with columns :code:`['patient_id', 'date', 'outcome']` and any additional 
        columns,  in which :code:`'patient_id'` refers
        to a column of str values, :code:`'date'` refers to a column of date values (accepted
        by :code:`pd.to_datetime`), and :code:`'outcome'` refers to a column of boolean values.

    - id_col:  str, optional:
        The column name in :code:`df` that contains the ID information.
        Defaults to :code:`'patient_id'`.

    - datetime_col:  str, optional:
        The column name in :code:`df` that contains the date time information.
        Defaults to :code:`'start_date'`.

    Returns
    ---------
    
    - df_labelled: pandas.DataFrame: 
        This is a dataframe containing the original data along with a new column, :code:`'outcome'`,
        which contains the labels.
    '''

    assert id_col in df.columns, 'Please ensure that the id_col is in the data frame. '\
        'You should specify id_col=[THE ID COLUMN]'
    assert datetime_col in df.columns, 'Please ensure that the datetime_col is in the data frame. '\
        'You should specify datetime_col=[THE DATE/DATETIME COLUMN]'

    df = df.copy()

    df_labels = label_function()

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
    df_labelled = pd.merge(df, df_labels, how='left', on=[id_col, '__date__']).copy().drop('__date__', axis=1).copy()

    return df_labelled








def _label_number_previous(
    df:pd.DataFrame, 
    label_function:typing.Callable, 
    id_col:str='patient_id', 
    datetime_col:str='start_date',
    day_delay:int=1,
    ):
    '''
    This function allows you to label the number of positives to date
    for the corresponding ID and date.
    

    Arguments
    ---------

    - df:  pandas.DataFrame: 
        The dataframe to append the number of previous label positives to.

    - label_function:  typing.Callable: 
        The labelling function to use. This should return the labels in a dataframe
        with columns :code:`['patient_id', 'date', 'outcome']`, in which :code:`'patient_id'` refers
        to a column of str values, :code:`'date'` refers to a column of date values (accepted
        by :code:`pd.to_datetime`), and :code:`'outcome'` refers to a column of boolean values.
    
    - id_col:  str, optional:
        The column name that contains the ID information.
        Defaults to :code:`'patient_id'`.

    - datetime_col:  str, optional:
        The column name that contains the date time information.
        Defaults to :code:`'start_date'`.

    - day_delay:  int, optional:
        The number of days after a label is detected when the data reflects
        that the ID has had another previous label. This is used to ensure
        that the predictive model does not simply learn that to look for 
        when this feature increases.
        Defaults to :code:`1`.

    
    Returns
    ---------
    
    - df_out: pandas.DataFrame: 
        This is a dataframe containing the original data along with a new column, :code:`f'previous_outcome'`,
        which contains the number of previous labels to date for that ID.

    '''

    assert id_col in df.columns, 'Please ensure that the id_col is in the data frame. '\
        'You should specify id_col=[THE ID COLUMN]'
    assert datetime_col in df.columns, 'Please ensure that the datetime_col is in the data frame. '\
        'You should specify datetime_col=[THE DATE/DATETIME COLUMN]'

    id_type = df[id_col].dtype

    cumulative_pos_labels = (label_function()
        .astype(object)
        .query('outcome == True')
        .groupby(['patient_id', 'date'])
        .sum()
        .groupby(level=0)
        .cumsum()
        .reset_index()
        .sort_values(by='date')
        .rename({'outcome': f'previous_outcome',
            'date': '__date__'}, axis=1)
        .astype({'patient_id': id_type})
        )

    cumulative_pos_labels['__date__'] = pd.to_datetime(
        cumulative_pos_labels['__date__']
        )

    
    # this ensures that the number of previous label increases the day_delay
    # after the label
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
    df[f'previous_outcome'] = df[f'previous_outcome'].fillna(0)

    return df

