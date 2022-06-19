import pandas as pd
import os
import sys
import numpy as np
import typing
import functools
from pandarallel import pandarallel as pandarallel_

import dcarte
from dcarte.utils import process_transition, localize_time
from dcarte.local import LocalDataset
from dcarte.config import get_config

from dcarte_transform.utils.progress import tqdm_style, pandarallel_progress
from dcarte_transform.transform.utils import datetime_compare_rolling, compute_delta, relative_func_delta
from dcarte_transform.transform.activity import compute_daily_location_freq, compute_entropy_rate

import logging
logging.warning('this is still under production')




@dcarte.utils.timer('processing sleep FE data')
def process_sleep(
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
def process_relative_transitions(
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

    # running calculations
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
                                .rename({datetime_col: 'date'}, axis=1))

    dtypes = {
                id_col: 'category',
                transition_col: 'category',
                'date': 'object',
                'source': 'category',
                'sink': 'category',
                }

    return daily_rel_transitions.astype(dtypes)







@dcarte.utils.timer('processing location frequency statistics')
def process_location_time_stats(
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
def process_entropy_data(
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





### still under production
if __name__ == '__main__':

    sleep = dcarte.load('sleep', 'base')
    sleep_stats = process_sleep(
                                sleep, 
                                id_col='patient_id', 
                                datetime_col='start_date',
                                )

    activity = dcarte.load('activity', 'raw')
    bathroom_freq_nighttime = process_location_time_stats(
                                                            activity, 
                                                            location_name='bathroom1', 
                                                            id_col='patient_id',
                                                            location_col='location_name',
                                                            datetime_col='start_date',
                                                            time_range=['20:00', '08:00'],
                                                            rolling_window=3,
                                                            name='bathroom_nighttime',
                                                            )

    bathroom_freq_daytime = process_location_time_stats(
                                                            activity, 
                                                            location_name='bathroom1', 
                                                            id_col='patient_id',
                                                            location_col='location_name',
                                                            datetime_col='start_date',
                                                            time_range=['08:00', '20:00'],
                                                            rolling_window=3,
                                                            name='bathroom_daytime',
                                                            )

    entropy_daily = process_entropy_data(
                                            activity, 
                                            freq='day',
                                            id_col='patient_id', 
                                            datetime_col='start_date',
                                            location_col='location_name'
                                            )


    relative_mean_delta = functools.partial(relative_func_delta, func=np.mean)
    relative_std_delta = functools.partial(relative_func_delta, func=np.std)

    transitions = dcarte.load('transitions', 'base')
    bathroom_relative_transitions = process_relative_transitions(
                                                        df=transitions,
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
                                            )