'''
Labelling agitation data
'''

import pandas as pd
import numpy as np
import dcarte
import datetime
import logging
import uuid
import functools
import typing

try:
    from ._base import _label, _label_number_previous
except:
    from dcarte_transform.label._base import _label, _label_number_previous


def resample_labels_database(database:pd.DataFrame, 
                             most_frequent_ids:pd.DataFrame):
    '''
        This function will return the expanded dataframe of the neuropsychiatric labels to include the false labels.

        Arguments
        ---------

        - database:  pd.DataFrame:
            Database to expande.

        - most_frequent_ids: pd.DataFrame:
            Dataframe including the ids of the participants suffering most frequently from neuropsychiatric symptoms.


        Returns
        --------

        - resampled_database:  pd.DataFrame` :
            A dataframe containing the expanded version of the neuropsychiatric labels, with the corresponding patient_id and
            date.

        '''
    resampled_database = pd.DataFrame()
    for patient in most_frequent_ids['patient_id']:
        current_data = database[database['patient_id'] == patient].reset_index(drop=True)
        if not current_data.empty:
            current_data = current_data.drop_duplicates(subset='start_date').set_index('start_date')
            new_index = pd.date_range(current_data.index.min(), current_data.index.max(), freq='D')
            current_data = current_data.reindex(new_index).reset_index()
            current_data['patient_id'] = patient
            current_data['type'] = database['type'].values[0]
            resampled_database = pd.concat([resampled_database, current_data])
    resampled_database = resampled_database.fillna(False)
    return resampled_database.reset_index(drop=True)


def get_labels(days_either_side:int=0, return_event:bool=False) -> pd.DataFrame:
    '''
    This function will return the Agitation labels.
    If a single day for a paticular ID contains two different
    labels (usually caused by using :code:`days_either_side`),
    then both labels are removed.

    Arguments
    ---------

    - days_either_side:  int, optional:
        The number of days either side of a label that will be given the same label.
        If these days overlap, if the label is the same then the first will be kept.
        If they are different, then neither will be kept.
        Defaults to :code:`0`.

    - return_event:  bool, optional:
        This dictates whether another column should be added, with a unique id given to each of the separate
        UTI events. This allows the user to group the outputted data based on events.
        Defaults to :code:`False`.


    Returns
    --------

    - out:  pd.DataFrame` :
        A dataframe containing the Agitation labels, with the corresponding patient_id and
        date.

    '''
    # extract the minder labels 
    symptoms = ['Depressed mood', 'Agitation', 'Anxiety', 'Irritability', 'Hallucinations', 'Delusions']

    minder_df = dcarte.load('Behavioural', 'RAW')
    minder_df = minder_df[minder_df['type'].isin(symptoms)].reset_index(drop=True)
    minder_df['notes'] = minder_df['notes'].str.lower()
    minder_df['label'] = np.nan
    minder_df['start_date'] = pd.to_datetime(minder_df['start_date']).dt.floor('D')
    minder_df['type'] = 'Neuropsychiatric' 

    minder_df.loc[minder_df['start_date']>'2022-03-01', 'label'] = 'positive'
    minder_df.loc[minder_df['notes'].str.contains('positive', na=False), 'label'] = 'positive'
    minder_df.loc[minder_df['notes'].str.contains('negative', na=False), 'label'] = 'negative'

    minder_df = minder_df.drop(columns=['home_id', 'sub_types', 'source', 'notes'])
    minder_df['label'] = minder_df['label'].replace('positive', True)
    minder_df['label'] = minder_df['label'].replace('negative', False)

    minder_df = minder_df.dropna()

    minder_df = minder_df.sort_values('label', ascending=False).drop_duplicates(subset=['patient_id', 'start_date'])
    minder_df = minder_df.sort_values(['patient_id', 'start_date']).reset_index(drop=True)

    # we don't need this anymore
    # most_frequent_ids = minder_df.groupby('patient_id').size().to_frame('size').reset_index().sort_values(by='size')
    # most_frequent_ids = most_frequent_ids[most_frequent_ids['size']>most_frequent_ids['size'].mean()]

    # minder_df = resample_labels_database(minder_df, most_frequent_ids) 
    # minder_df.rename(columns={'index': 'start_date'}, inplace=True)

    # add the care questionnaire results
    questions = ['weekly-checking-PLWDAgitation-boolean',
             'weekly-checking-PLWDConfusion-boolean']

    questionnaire = dcarte.load('Questionnaire', 'CARE')
    questionnaire = questionnaire[questionnaire['question'].isin(questions)].reset_index(drop=True)
    questionnaire['question'].replace({'weekly-checking-PLWDAgitation-boolean': 'Neuropsychiatric', 'weekly-checking-PLWDConfusion-boolean': 'Neuropsychiatric'}, inplace=True)
    questionnaire['authored'] = questionnaire['authored'].dt.floor('D')
    questionnaire['answer'] = questionnaire['answer'].astype('float')
    questionnaire['answer'] = questionnaire['answer'].replace({0.0:False, 1.0:True})
    questionnaire.rename(columns={'question': 'type', 'answer': 'label', 'authored': 'start_date'}, inplace=True)
    questionnaire.drop(columns=['response_id', 'questionnaire', 'source'], inplace=True)
    minder_df = pd.concat([minder_df, questionnaire])


    # include also the TIHM labels
    tihm_df = dcarte.load('Flags', 'LEGACY')
    flags_columns_to_keep = ['subjectdf', 'type', 'datetimeRaised', 'valid']
    tihm_df = tihm_df[flags_columns_to_keep]
    tihm_df = tihm_df.rename(columns={'datetimeRaised': 'start_date', 'subjectdf': 'patient_id', 'valid': 'label'})

    tihm_df = tihm_df[tihm_df['type'].isin(symptoms)]
    tihm_df['type'] = 'Neuropsychiatric'
    tihm_df['start_date'] = pd.to_datetime(tihm_df['start_date']).dt.floor('D')

    df_labels = pd.concat([tihm_df, minder_df]).drop(columns='type').drop_duplicates().reset_index(drop=True)
    df_labels = df_labels.rename(columns={'start_date': 'date', 'label': 'agitation_label'})
    df_labels['date'] = pd.to_datetime(df_labels['date']).dt.date

    if return_event:
        df_labels['event'] = [uuid.uuid4() for _ in range(df_labels.shape[0])]

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

        groupby_names = (['patient_id', 'date', 'event', 'agitation_label'] if return_event
                         else ['patient_id', 'date', 'agitation_label'])
        df_labels = df_labels.groupby(groupby_names).apply(dates_either_side_group_by).reset_index(drop=True)
    else:
        df_labels.reset_index(drop=True)

    # removing the rows with contradictory labels
    df_labels = (df_labels
                    # keeping one of the rows where the labels are the same
                    .drop_duplicates(subset=['patient_id', 'date', 'agitation_label'], keep='first')
                    # removing rows where the labels are the different
                    .drop_duplicates(subset=['patient_id', 'date'], keep=False))

    return df_labels.reset_index(drop=True).dropna(subset='agitation_label')





@dcarte.utils.timer('mapping Agitation labels')
def label(
    df:pd.DataFrame, 
    id_col:str='patient_id', 
    datetime_col:str='start_date', 
    days_either_side:int=0, 
    return_event:bool=False,
    ) -> pd.DataFrame:
    '''
    This function will label the input dataframe based on the agitation data 
    in :code:`behaviour`.

    Arguments
    ----------
    
    - df:  pandas.DataFrame: 
        Unlabelled dataframe, must contain columns :code:`[id_col, datetime_col]`, where :code:`id_col` is the
        ids of participants and :code:`datetime_col` is the time of the sensors.

    - id_col:  str, optional:
        The column name that contains the ID information.
        Defaults to :code:`'patient_id'`.

    - datetime_col:  str, optional:
        The column name that contains the date time information.
        Defaults to :code:`'start_date'`.

    - days_either_side:  int, optional:
        The number of days either side of a label that will be given the same label.
        Defaults to :code:`0`.
    
    - return_event:  bool, optional:
        This dictates whether another column should be added, with a unique id given to each of the separate
        agitation events. This allows the user to group the outputted data based on events.
        Defaults to :code:`False`.

    Returns
    ---------
    
    - df_labelled: pandas.DataFrame: 
        This is a dataframe containing the original data along with a new column, :code:`'agitation_labels'`,
        which contains the labels. If :code:`return_event=True`, a column titled :code:`'agitation_event'` will be 
        added which contains unique IDs for each of the agitation episodes.
    '''

    assert type(days_either_side) == int, 'days_either_side must be an integer.'

    df_out = _label(
        df=df, 
        label_function=functools.partial(
            get_labels, 
            days_either_side=days_either_side, 
            return_event=return_event,
            ),
        id_col=id_col,
        datetime_col=datetime_col,
        )
    return df_out.rename(columns={'event': 'agitation_event'})


if __name__=='__main__':
    # extract just the labels
    labels = get_labels() 
    print('Positive Lables : {} - Negative Labels : {}'.format( np.sum(labels['agitation_label']==True), np.sum(labels['agitation_label']==False)))
    
    # label a specific dataset
    data = dcarte.load('activity', 'raw')
    data_labelled = label(data, days_either_side=2, return_event=True)
    n_positives = np.sum(data_labelled['agitation_label']==True)
    n_negatives = np.sum(data_labelled['agitation_label']==False)
    print(f'Agitation: There are {n_positives} positively and {n_negatives} negatively labelled rows.')
    