import pandas as pd
import typing
import functools
from pandarallel import pandarallel as pandarallel_

import dcarte
from dcarte.local import LocalDataset

from dcarte_transform.utils.progress import tqdm_style, pandarallel_progress
from dcarte_transform.transform.utils import datetime_compare_rolling, compute_delta
from dcarte_transform.transform.activity import compute_daily_location_freq, compute_entropy_rate



########## computing functions

@dcarte.utils.timer('processing sleep FE data')
def compute_sleep(
                    df:pd.DataFrame, 
                    id_col:str,
                    datetime_col:str,
                    ):
    df = df.copy()

    def count_helper(name_to_count):
        def count(x):
            return x[x==name_to_count].shape[0]
        count.__name__ = f'count_{name_to_count}'
        return count

    df_grouped = df.groupby(by=[id_col, 
                        pd.Grouper(key=datetime_col, freq='1d')]).agg({
                            'snoring': 'sum',
                            'heart_rate': ['mean', 'std'],
                            'respiratory_rate': ['mean', 'std'],
                            'state': [
                                count_helper('AWAKE'), 
                                count_helper('DEEP'), 
                                count_helper('LIGHT'), 
                                count_helper('REM')
                                ],
                            }).reset_index(drop=False)

    df_grouped.columns = df_grouped.columns.map('|'.join).str.strip('|')
    df_grouped = df_grouped.rename({datetime_col: 'date'}, axis=1)
    df_grouped['date'] = df_grouped['date'].dt.date

    dtypes = {
                id_col: 'category',
                'date': 'object',
                'snoring|sum': 'int',
                'heart_rate|mean': 'float',
                'heart_rate|std': 'float',
                'respiratory_rate|mean': 'float', 
                'respiratory_rate|std': 'float',
                'state|count_AWAKE': 'int', 
                'state|count_DEEP': 'int',
                'state|count_LIGHT': 'int',
                'state|count_REM': 'int',
                }

    return df_grouped.astype(dtypes)










@dcarte.utils.timer('processing relative_transition data')
def compute_relative_transitions(
                                    df:pd.DataFrame, 
                                    funcs:typing.Union[typing.List[typing.Callable], typing.Callable],
                                    id_col:str,
                                    transition_col:str,
                                    datetime_col:str,
                                    duration_col:str,
                                    sink_col:str,
                                    source_col:str,
                                    filter_sink:typing.Union[None,typing.List[str]]=None, 
                                    filter_source:typing.Union[None,typing.List[str]]=None,
                                    transition_time_upper_bound:float=5*60,
                                    s='1d', 
                                    w_distribution='7d', 
                                    w_sample='1d', 
                                    ):
    df = df.copy()

    # filtering on transition time upper bound
    df=df[df[duration_col]<transition_time_upper_bound]

    # filtering the sink values
    if not filter_sink is None:
        if type(filter_sink) == str:
            filter_sink = [filter_sink]
        df = df[df[sink_col].isin(filter_sink)] 

    # filtering the source values
    if not filter_source is None:
        if type(filter_source) == str:
            filter_source = [filter_source]
        df = df[df[source_col].isin(filter_source)] 

    df[source_col] = df[source_col].cat.remove_unused_categories()
    df[sink_col] = df[sink_col].cat.remove_unused_categories()

    if not type(funcs) == list:
        funcs = [funcs]

    # setting up parallel compute
    pandarallel_progress(desc="Computing transition function deltas", smoothing=0, **tqdm_style)
    pandarallel_.initialize(progress_bar=True, verbose=0)

    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # setting up arguments for the rolling calculations
    datetime_compare_rolling_partial = functools.partial(
                                        datetime_compare_rolling, 
                                        funcs=funcs, 
                                        s=s, 
                                        w_distribution=w_distribution, 
                                        w_sample=w_sample, 
                                        datetime_col=datetime_col, 
                                        value_col=duration_col,
                                        label='left',
                                        )

    # running calculations to calculate the func over each w_sample of an events
    # and the w_distribution of the same events.
    # if func is relative mean delta then this will calculate the change in mean 
    # of an event from one day to the past week.
    daily_rel_transitions = (df
                            [[id_col, transition_col, source_col, sink_col, datetime_col, duration_col]]
                            .sort_values(datetime_col)
                            .dropna()
                            .groupby(by=[id_col, source_col, sink_col, transition_col,])
                            .parallel_apply(datetime_compare_rolling_partial)
                            #.apply(datetime_compare_rolling_partial)
                        )

    # formatting
    daily_rel_transitions[datetime_col] = pd.to_datetime(daily_rel_transitions[datetime_col]).dt.date
    daily_rel_transitions = (daily_rel_transitions
                                .reset_index(drop=False)
                                .drop(['level_4',], axis=1)
                                .rename({datetime_col: 'date'}, axis=1)
                                .dropna()
                                )

    dtypes = {
                id_col: 'category',
                transition_col: 'category',
                'date': 'object',
                'source': 'category',
                'sink': 'category',
                }

    return daily_rel_transitions.astype(dtypes)







@dcarte.utils.timer('processing location frequency statistics')
def compute_location_time_stats(
                                    df:pd.DataFrame, 
                                    location_name:str, 
                                    id_col:str,
                                    location_col:str,
                                    datetime_col:str,
                                    time_range:typing.Union[None, typing.List[str]]=None, 
                                    rolling_window:int=3,
                                    name:typing.Union[str, None]=None,
                                    ):
    df = df.copy()

    if name is None:
        name = f'{location_name}'

    # computing the location frequency for the given location
    df = compute_daily_location_freq(df, 
                                        location=location_name, 
                                        id_col=id_col,
                                        location_col=location_col,
                                        datetime_col=datetime_col,
                                        time_range=time_range, 
                                        name=f'{name}_freq')

    # computing moving average
    df[f'{name}_freq_ma'] = (df
                                [[f'{name}_freq']]
                                .rolling(rolling_window)
                                .mean())

    # computing the delta in the moving average
    df[f'{name}_freq_ma_delta'] = compute_delta(df[f'{name}_freq_ma'].values,
                                                            pad=True
                                                            )

    # formatting
    dtypes = {
                id_col: 'category',
                'date': 'object',
                f'{name}_freq': 'int',
                f'{name}_freq_ma': 'float',
                f'{name}_freq_ma_delta': 'float',
                }

    return df.astype(dtypes)








@dcarte.utils.timer('processing entropy')
def compute_entropy_data(
                            df:pd.DataFrame, 
                            freq:str,
                            id_col:str,
                            datetime_col:str,
                            location_col:str,
                            ):
    df = df.copy()

    assert freq in ['day','week'], "Please ensure that freq is a string, either 'day' or 'week'"
    entropy_df = compute_entropy_rate(df, 
                                        freq=freq, 
                                        id_col=id_col, 
                                        datetime_col=datetime_col, 
                                        location_col=location_col,
                                        )
    entropy_col_name = 'daily_entropy' if freq == 'day' else 'weekly_entropy'

    dtypes = {
                id_col: 'category',
                'date': 'object',
                entropy_col_name: 'float',
                }

    return entropy_df.astype(dtypes)







########## processing functions




def process_sleep(self):
    df = self.datasets['sleep']
    sleep_stats = compute_sleep(
                            df, 
                            id_col='patient_id', 
                            datetime_col='start_date',
                            )
    return sleep_stats





def process_relative_transitions(self):

    df = self.datasets['transitions']

    def relative_mean_delta(array_distribution, array_sample):
        # imports are required for parralisation on windows
        from dcarte_transform.transform.utils import relative_func_delta
        import numpy as np
        return relative_func_delta(array_distribution, array_sample, func=np.mean)

    def relative_std_delta(array_distribution, array_sample):
        # imports are required for parralisation on windows
        from dcarte_transform.transform.utils import relative_func_delta
        import numpy as np
        return relative_func_delta(array_distribution, array_sample, func=np.std)


    bathroom_relative_transitions = compute_relative_transitions(
                                                        df=df,
                                                        funcs=[
                                                                relative_mean_delta,
                                                                relative_std_delta
                                                                ],
                                                        id_col='patient_id',
                                                        transition_col='transition',
                                                        datetime_col='start_date',
                                                        duration_col='dur',
                                                        sink_col='sink',
                                                        source_col='source',
                                                        filter_sink='Bathroom',
                                                        filter_source=None,
                                                        transition_time_upper_bound=5*60,
                                                        s='1d',
                                                        w_distribution='7d',
                                                        w_sample='1d',
                                                        )


    bathroom_relative_transitions = (bathroom_relative_transitions
                                            .groupby(by=['patient_id', 'sink', 'date'])
                                            .mean()
                                            .reset_index(drop=False)
                                            .dropna()
                                            .drop(['sink'], axis=1)
                                            )
    
    return bathroom_relative_transitions





def process_bathroom_nighttime_stats(self):

    df = self.datasets['activity']
    bathroom_freq_nighttime = compute_location_time_stats(
                                                            df, 
                                                            location_name='bathroom1', 
                                                            id_col='patient_id',
                                                            location_col='location_name',
                                                            datetime_col='start_date',
                                                            time_range=['20:00', '08:00'],
                                                            rolling_window=3,
                                                            name='bathroom_nighttime',
                                                            )
    return bathroom_freq_nighttime





def process_bathroom_daytime_stats(self):

    df = self.datasets['activity']
    bathroom_freq_daytime = compute_location_time_stats(
                                                            df, 
                                                            location_name='bathroom1', 
                                                            id_col='patient_id',
                                                            location_col='location_name',
                                                            datetime_col='start_date',
                                                            time_range=['08:00', '20:00'],
                                                            rolling_window=3,
                                                            name='bathroom_daytime',
                                                            )
    return bathroom_freq_daytime





def process_entropy_daily(self):
    
    df = self.datasets['activity']
    entropy_daily = compute_entropy_data(
                                            df, 
                                            freq='day',
                                            id_col='patient_id', 
                                            datetime_col='start_date',
                                            location_col='location_name'
                                            )
    
    return entropy_daily







def process_fe_data(self):

    sleep_fe = self.datasets['sleep_fe']
    bathroom_nighttime_fe = self.datasets['bathroom_nighttime_fe']
    bathroom_daytime_fe = self.datasets['bathroom_daytime_fe']
    entropy_daily_fe = self.datasets['entropy_daily_fe']
    bathroom_relative_transitions_fe = self.datasets['bathroom_relative_transitions_fe']

    fe_data = pd.merge(left=sleep_fe, right=bathroom_nighttime_fe, on=['patient_id', 'date'], how='outer')
    fe_data = pd.merge(left=fe_data, right=bathroom_daytime_fe, on=['patient_id', 'date'], how='outer')
    fe_data = pd.merge(left=fe_data, right=entropy_daily_fe, on=['patient_id', 'date'], how='outer')
    fe_data = pd.merge(left=fe_data, right=bathroom_relative_transitions_fe, on=['patient_id', 'date'], how='outer')

    return fe_data








########## creating datasets

def create_feature_engineering_datasets():
    domain = 'feature_engineering'
    module = 'feature_engineering'
    # since = '2022-02-10'
    # until = '2022-02-20'
    parent_datasets = {'sleep_fe': [['sleep', 'base']],
                        'bathroom_relative_transitions_fe':[['transitions', 'base']],
                        'bathroom_nighttime_fe': [['activity', 'raw']],
                        'bathroom_daytime_fe': [['activity', 'raw']],
                        'entropy_daily_fe': [['activity', 'raw']],
                        'all_fe': [['sleep_fe', 'feature_engineering'],
                                ['bathroom_relative_transitions_fe', 'feature_engineering'],
                                ['bathroom_nighttime_fe', 'feature_engineering'],
                                ['bathroom_daytime_fe', 'feature_engineering'],
                                ['entropy_daily_fe', 'feature_engineering'],
                                ]
                        }

    module_path = __file__

    print('processing sleep FE')
    LocalDataset(dataset_name='sleep_fe',
                 datasets={d[0]: dcarte.load(*d) for d in parent_datasets['sleep_fe']},
                 pipeline=['process_sleep'],
                 domain=domain,
                 module=module,
                 module_path=module_path,
                 reload=True,
                 dependencies=parent_datasets['sleep_fe'])

    print('processing night time bathroom FE')
    LocalDataset(dataset_name='bathroom_nighttime_fe',
                 datasets={d[0]: dcarte.load(*d) for d in parent_datasets['bathroom_nighttime_fe']},
                 pipeline=['process_bathroom_nighttime_stats'],
                 domain=domain,
                 module=module,
                 reload=True,
                 module_path=module_path,
                 dependencies=parent_datasets['bathroom_nighttime_fe'])

    print('processing day time bathroom FE')
    LocalDataset(dataset_name='bathroom_daytime_fe',
                 datasets={d[0]: dcarte.load(*d) for d in parent_datasets['bathroom_daytime_fe']},
                 pipeline=['process_bathroom_daytime_stats'],
                 domain=domain,
                 module=module,
                 reload=True,
                 module_path=module_path,
                 dependencies=parent_datasets['bathroom_daytime_fe'])

    print('processing daily entropy FE')
    LocalDataset(dataset_name='entropy_daily_fe',
                 datasets={d[0]: dcarte.load(*d) for d in parent_datasets['entropy_daily_fe']},
                 pipeline=['process_entropy_daily'],
                 domain=domain,
                 module=module,
                 reload=True,
                 module_path=module_path,
                 dependencies=parent_datasets['entropy_daily_fe'])


    print('processing bathroom transition FE')
    LocalDataset(dataset_name='bathroom_relative_transitions_fe',
                 datasets={d[0]: dcarte.load(*d) for d in parent_datasets['bathroom_relative_transitions_fe']},
                 pipeline=['process_relative_transitions'],
                 domain=domain,
                 module=module,
                 reload=True,
                 module_path=module_path,
                 dependencies=parent_datasets['bathroom_relative_transitions_fe'])



    print('processing all FE')
    LocalDataset(dataset_name='all_fe',
                 datasets={d[0]: dcarte.load(*d) for d in parent_datasets['all_fe']},
                 pipeline=['process_fe_data'],
                 domain=domain,
                 module=module,
                 module_path=module_path,
                 reload=True,
                 dependencies=parent_datasets['all_fe'])






if __name__ == '__main__':
    create_feature_engineering_datasets()