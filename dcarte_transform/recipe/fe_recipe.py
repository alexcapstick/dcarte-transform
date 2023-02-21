"""
Feature engineering recipe.
"""

import pandas as pd
import numpy as np
import typing
import functools
import dcarte
from dcarte.local import LocalDataset

try:
    from pandarallel import pandarallel as pandarallel_
    pandarallel_import_error = False
except ImportError:
    pandarallel_import_error = True

try:
    from dcarte_transform.utils.progress import tqdm_style, pandarallel_progress
    from dcarte_transform.transform.utils import datetime_compare_rolling, compute_delta
    from dcarte_transform.transform.activity import (
        compute_daily_location_freq,
        compute_entropy_rate,
        fill_from_first_occurence,
    )
    from dcarte_transform.recipe.tihm_and_minder_recipe import (
        create_tihm_and_minder_datasets,
    )
    from dcarte_transform.label.uti import label_number_previous
except ImportError:
    from ..utils.progress import tqdm_style, pandarallel_progress
    from ..transform.utils import datetime_compare_rolling, compute_delta
    from ..transform.activity import (
        compute_daily_location_freq,
        compute_entropy_rate,
        fill_from_first_occurence,
    )
    from ..recipe.tihm_and_minder_recipe import create_tihm_and_minder_datasets
    from ..label.uti import label_number_previous

######### More readable feature names

feature_nice_names = {
    "bathroom_daytime_freq": "Bathroom Daytime Frequency",
    "bathroom_daytime_freq_ma": "Bathroom Moving Average Daytime Frequency",
    "bathroom_daytime_freq_ma_delta": "Change in Bathroom Moving Average Daytime Frequency",
    "bathroom_nighttime_freq": "Bathroom Nighttime Frequency",
    "bathroom_nighttime_freq_ma": "Bathroom Moving Average Nighttime Frequency",
    "bathroom_nighttime_freq_ma_delta": "Change in Bathroom Moving Average Nighttime Frequency",
    "bathroom_relative_transition_time_delta_mean": "Mean of the Relative Transition Time to the Bathroom",
    "bathroom_relative_transition_time_delta_std": "Standard Deviation of the Relative Transition Time to the Bathroom",
    "daily_entropy": "Daily Entropy of Movement",
    "freq|Bathroom": "Frequency of Visits to the Bathroom",
    "freq|Bedroom": "Frequency of Visits to the Bedroom",
    "freq|Hallway": "Frequency of Visits to the Hallway",
    "freq|Kitchen": "Frequency of Visits to the Kitchen",
    "freq|Lounge": "Frequency of Visits to the Lounge",
    "heart_rate|mean": "Mean of the Nighttime Heart Rate",
    "heart_rate|std": "Standard Deviation of the Nighttime Heart Rate",
    "respiratory_rate|mean": "Mean of the Nighttime Respiratory Rate",
    "respiratory_rate|std": "Standard Deviation of the Nighttime Respiratory Rate",
    "state|count_AWAKE": "Number of Nighttime Awakenings",
    "previous_uti": "Number of Previous UTIs",
}


CORE_LOCATIONS = [
    "Bathroom",
    "Bedroom",
    "Hallway",
    "Kitchen",
    "Lounge",
]

PREVIOUS_UTI_DELAY = 21

########## computing functions


@dcarte.utils.timer("processing sleep data")
def compute_sleep(
    sleep: pd.DataFrame,
    id_col: str,
    datetime_col: str,
):

    sleep = sleep.copy()
    sleep = sleep.assign(start_date=lambda x: pd.to_datetime(x["start_date"]))

    # sleep states

    sleep_states = sleep["state"].unique()
    sleep_states_df = (
        sleep[[id_col, datetime_col, "state"]]
        .groupby(by=[id_col, pd.Grouper(key=datetime_col, freq="1d"), "state"])
        .size()
        .to_frame(name="freq")
        .unstack()
        .reset_index()
        .sort_values(["patient_id", "start_date"])
    )

    sleep_states_df.columns = sleep_states_df.columns.map("|".join).str.strip("|")
    # making sure that all 0s are NaNs are correct imputing
    sleep_states_df = sleep_states_df.replace(
        {f"freq|{state}": {0: np.nan} for state in sleep_states}
    )
    # ensuring that if there is at least one sleep state for the day, then nan is 0
    sleep_states_only_df = sleep_states_df[[f"freq|{state}" for state in sleep_states]]
    sleep_states_only_imputed = sleep_states_only_df[
        pd.isna(sleep_states_only_df)
        .sum(axis=1)
        .between(0, len(sleep_states), inclusive="neither")
    ].fillna(0)
    # removing the sleep days in which there are no recorded states
    sleep_states_df.loc[
        sleep_states_only_imputed.index, sleep_states_only_imputed.columns
    ] = sleep_states_only_imputed
    sleep_states_df = sleep_states_df[
        (~(pd.isna(sleep_states_only_df).sum(axis=1) == len(sleep_states)).values)
    ]

    sleep_states_df["date"] = pd.to_datetime(sleep_states_df["start_date"]).dt.date
    sleep_states_df = sleep_states_df.drop("start_date", axis=1)

    # sleep physio

    sleep_physio_df = (
        sleep[[id_col, datetime_col, "heart_rate", "respiratory_rate"]]
        .groupby(by=[id_col, pd.Grouper(key=datetime_col, freq="1d")])
        .agg(
            {
                "heart_rate": ["mean", "std"],
                "respiratory_rate": ["mean", "std"],
            }
        )
    )
    # removing those days in which there is no data
    sleep_physio_df = sleep_physio_df[~pd.isna(sleep_physio_df).all(axis=1).values]
    sleep_physio_df = sleep_physio_df.reset_index()
    sleep_physio_df.columns = sleep_physio_df.columns.map("|".join).str.strip("|")
    sleep_physio_df["date"] = pd.to_datetime(sleep_physio_df["start_date"]).dt.date
    sleep_physio_df = sleep_physio_df.drop("start_date", axis=1)

    return pd.merge(
        sleep_states_df, sleep_physio_df, on=["patient_id", "date"], how="outer"
    )


@dcarte.utils.timer("processing relative_transition data")
def compute_relative_transitions(
    df: pd.DataFrame,
    funcs: typing.Union[typing.List[typing.Callable], typing.Callable],
    id_col: str,
    transition_col: str,
    datetime_col: str,
    duration_col: str,
    sink_col: str,
    source_col: str,
    filter_sink: typing.Union[None, typing.List[str]] = None,
    filter_source: typing.Union[None, typing.List[str]] = None,
    transition_time_upper_bound: float = 5 * 60,
    s="1d",
    w_distribution="7d",
    w_sample="1d",
):

    if pandarallel_import_error:
        raise ImportError(
            "pandarallel is not installed. Please install it to use this function."
        )

    df = df.copy()

    # filtering on transition time upper bound
    df = df[df[duration_col] < transition_time_upper_bound]

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
    pandarallel_progress(
        desc="Computing transition function deltas", smoothing=0, **tqdm_style
    )
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
        label="left",
    )

    # running calculations to calculate the func over each w_sample of an events
    # and the w_distribution of the same events.
    # if func is relative mean delta then this will calculate the change in mean
    # of an event from one day to the past week.
    daily_rel_transitions = (
        df[[id_col, transition_col, source_col, sink_col, datetime_col, duration_col]]
        .sort_values(datetime_col)
        .dropna()
        .groupby(
            by=[
                id_col,
                source_col,
                sink_col,
                transition_col,
            ]
        )
        .parallel_apply(datetime_compare_rolling_partial)
        # .apply(datetime_compare_rolling_partial)
    )

    # formatting
    daily_rel_transitions[datetime_col] = pd.to_datetime(
        daily_rel_transitions[datetime_col]
    ).dt.date
    daily_rel_transitions = (
        daily_rel_transitions.reset_index(drop=False)
        .drop(
            [
                "level_4",
            ],
            axis=1,
        )
        .rename({datetime_col: "date"}, axis=1)
        .dropna()
    )

    dtypes = {
        id_col: "category",
        transition_col: "category",
        "date": "object",
        "source": "category",
        "sink": "category",
    }

    return daily_rel_transitions.astype(dtypes)


@dcarte.utils.timer("processing location frequency statistics")
def compute_location_time_stats(
    df: pd.DataFrame,
    location_name: str,
    id_col: str,
    location_col: str,
    datetime_col: str,
    time_range: typing.Union[None, typing.List[str]] = None,
    rolling_window: int = 3,
    name: typing.Union[str, None] = None,
):
    df = df.copy()

    if name is None:
        name = f"{location_name}".lower()

    # computing the location frequency for the given location
    df = compute_daily_location_freq(
        df,
        location=location_name,
        id_col=id_col,
        location_col=location_col,
        datetime_col=datetime_col,
        time_range=time_range,
        name=f"{name}_freq",
    )

    # computing moving average
    df[f"{name}_freq_ma"] = df[[f"{name}_freq"]].rolling(rolling_window).mean()

    # computing the delta in the moving average
    df[f"{name}_freq_ma_delta"] = compute_delta(df[f"{name}_freq_ma"].values, pad=True)

    # formatting
    dtypes = {
        id_col: "category",
        "date": "object",
        f"{name}_freq": "int",
        f"{name}_freq_ma": "float",
        f"{name}_freq_ma_delta": "float",
    }

    return df.astype(dtypes)


@dcarte.utils.timer("processing entropy")
def compute_entropy_data(
    df: pd.DataFrame,
    freq: str,
    id_col: str,
    datetime_col: str,
    location_col: str,
):
    df = df.copy()

    assert freq in [
        "day",
        "week",
    ], "Please ensure that freq is a string, either 'day' or 'week'"
    entropy_df = compute_entropy_rate(
        df,
        freq=freq,
        id_col=id_col,
        datetime_col=datetime_col,
        location_col=location_col,
    )
    entropy_col_name = "daily_entropy" if freq == "day" else "weekly_entropy"

    dtypes = {
        id_col: "category",
        "date": "object",
        entropy_col_name: "float",
    }

    return entropy_df.astype(dtypes)


########## processing functions


def process_sleep(self):
    df = self.datasets["sleep"]
    sleep_stats = compute_sleep(
        df,
        id_col="patient_id",
        datetime_col="start_date",
    )
    return sleep_stats


def process_relative_transitions(self):

    df = self.datasets["transitions"]

    def relative_mean_delta(array_sample, array_distribution):
        # imports are required for parralisation on windows
        from dcarte_transform.transform.utils import relative_func_delta
        import numpy as np

        return relative_func_delta(array_sample, array_distribution, func=np.mean)

    def relative_std_delta(array_sample, array_distribution):
        # imports are required for parralisation on windows
        from dcarte_transform.transform.utils import relative_func_delta
        import numpy as np

        return relative_func_delta(array_sample, array_distribution, func=np.std)

    bathroom_relative_transitions = compute_relative_transitions(
        df=df,
        funcs=[relative_mean_delta, relative_std_delta],
        id_col="patient_id",
        transition_col="transition",
        datetime_col="start_date",
        duration_col="dur",
        sink_col="sink",
        source_col="source",
        filter_sink="Bathroom",
        filter_source=None,
        transition_time_upper_bound=5 * 60,
        s="1d",
        w_distribution="7d",
        w_sample="1d",
    )

    bathroom_relative_transitions = (
        bathroom_relative_transitions.groupby(by=["patient_id", "sink", "date"])
        .mean()
        .reset_index(drop=False)
        .dropna()
        .drop(["sink"], axis=1)
        .rename(
            {
                "relative_mean_delta_relative_delta": "bathroom_relative_transition_time_delta_mean",
                "relative_std_delta_relative_delta": "bathroom_relative_transition_time_delta_std",
            },
            axis=1,
        )
    )

    return bathroom_relative_transitions


def process_bathroom_nighttime_stats(self):

    df = self.datasets["motion"]
    bathroom_freq_nighttime = compute_location_time_stats(
        df,
        location_name="Bathroom",
        id_col="patient_id",
        location_col="location_name",
        datetime_col="start_date",
        time_range=["20:00", "08:00"],
        rolling_window=3,
        name="bathroom_nighttime",
    )
    return bathroom_freq_nighttime


def process_bathroom_daytime_stats(self):

    df = self.datasets["motion"]
    bathroom_freq_daytime = compute_location_time_stats(
        df,
        location_name="Bathroom",
        id_col="patient_id",
        location_col="location_name",
        datetime_col="start_date",
        time_range=["08:00", "20:00"],
        rolling_window=3,
        name="bathroom_daytime",
    )
    return bathroom_freq_daytime


def process_entropy_daily(self):

    df = self.datasets["motion"]
    entropy_daily = compute_entropy_data(
        df,
        freq="day",
        id_col="patient_id",
        datetime_col="start_date",
        location_col="location_name",
    )

    return entropy_daily


def process_fe_data(self):

    sleep_fe = self.datasets["sleep_fe"]
    bathroom_nighttime_fe = self.datasets["bathroom_nighttime_fe"]
    bathroom_daytime_fe = self.datasets["bathroom_daytime_fe"]
    entropy_daily_fe = self.datasets["entropy_daily_fe"]
    bathroom_relative_transitions_fe = self.datasets["bathroom_relative_transitions_fe"]

    fe_data = pd.merge(
        left=sleep_fe,
        right=bathroom_nighttime_fe,
        on=["patient_id", "date"],
        how="outer",
    )
    fe_data = pd.merge(
        left=fe_data,
        right=bathroom_daytime_fe,
        on=["patient_id", "date"],
        how="outer",
    )
    fe_data = pd.merge(
        left=fe_data,
        right=entropy_daily_fe,
        on=["patient_id", "date"],
        how="outer",
    )
    fe_data = pd.merge(
        left=fe_data,
        right=bathroom_relative_transitions_fe,
        on=["patient_id", "date"],
        how="outer",
    )

    fe_data = label_number_previous(
        fe_data,
        id_col="patient_id",
        datetime_col="date",
        day_delay=PREVIOUS_UTI_DELAY,
    )

    return fe_data


def process_core_raw_data(self):
    motion = self.datasets["motion"]
    motion = motion[["patient_id", "start_date", "location_name"]]

    motion = (
        motion.assign(start_date=lambda x: pd.to_datetime(x["start_date"]))
        .query("location_name in @CORE_LOCATIONS")
        .groupby(
            by=[
                "patient_id",
                pd.Grouper(key="start_date", freq="1d"),
                "location_name",
            ]
        )
        .size()
        .to_frame(name="freq")
        .unstack()
        .reset_index()
        .sort_values(["patient_id", "start_date"])
    )

    motion.columns = motion.columns.map("|".join).str.strip("|")

    # making sure that all 0s are NaNs are correct imputing
    motion = motion.replace({f"freq|{loc}": {0: np.nan} for loc in CORE_LOCATIONS})

    motion = motion.groupby("patient_id", group_keys=False).apply(
        lambda x: fill_from_first_occurence(
            x, subset=[f"freq|{loc}" for loc in CORE_LOCATIONS], value=0
        )
    )

    motion["date"] = pd.to_datetime(motion["start_date"]).dt.date
    motion = motion.drop("start_date", axis=1)

    return motion


def process_core_raw_and_fe_data(self):
    motion = self.datasets["core_raw"]
    fe_data = self.datasets["all_fe"]

    core_raw_fe_data = pd.merge(
        motion,
        fe_data,
        on=["patient_id", "date"],
        how="outer",
    )

    core_raw_fe_data = core_raw_fe_data.drop("previous_uti", axis=1)
    core_raw_fe_data = label_number_previous(
        core_raw_fe_data,
        id_col="patient_id",
        datetime_col="date",
        day_delay=PREVIOUS_UTI_DELAY,
    )

    return core_raw_fe_data


########## creating datasets


def create_feature_engineering_datasets():
    domain = "feature_engineering"
    module = "feature_engineering"
    # since = '2022-02-10'
    # until = '2022-02-20'

    if not "TIHM_AND_MINDER" in dcarte.domains().columns:
        create_tihm_and_minder_datasets()

    parent_datasets = {
        "sleep_fe": [["sleep", "base"]],
        "bathroom_relative_transitions_fe": [["transitions", "tihm_and_minder"]],
        "bathroom_nighttime_fe": [["motion", "tihm_and_minder"]],
        "bathroom_daytime_fe": [["motion", "tihm_and_minder"]],
        "entropy_daily_fe": [["motion", "tihm_and_minder"]],
        "all_fe": [
            ["sleep_fe", "feature_engineering"],
            ["bathroom_relative_transitions_fe", "feature_engineering"],
            ["bathroom_nighttime_fe", "feature_engineering"],
            ["bathroom_daytime_fe", "feature_engineering"],
            ["entropy_daily_fe", "feature_engineering"],
        ],
        "core_raw": [["motion", "tihm_and_minder"]],
        "all_core_raw_fe": [
            ["core_raw", "feature_engineering"],
            ["all_fe", "feature_engineering"],
        ],
    }

    module_path = __file__

    print("processing sleep FE")
    LocalDataset(
        dataset_name="sleep_fe",
        datasets={d[0]: dcarte.load(*d) for d in parent_datasets["sleep_fe"]},
        pipeline=["process_sleep"],
        domain=domain,
        module=module,
        module_path=module_path,
        reload=True,
        dependencies=parent_datasets["sleep_fe"],
    )

    print("processing night time bathroom FE")
    LocalDataset(
        dataset_name="bathroom_nighttime_fe",
        datasets={
            d[0]: dcarte.load(*d) for d in parent_datasets["bathroom_nighttime_fe"]
        },
        pipeline=["process_bathroom_nighttime_stats"],
        domain=domain,
        module=module,
        reload=True,
        module_path=module_path,
        dependencies=parent_datasets["bathroom_nighttime_fe"],
    )

    print("processing day time bathroom FE")
    LocalDataset(
        dataset_name="bathroom_daytime_fe",
        datasets={
            d[0]: dcarte.load(*d) for d in parent_datasets["bathroom_daytime_fe"]
        },
        pipeline=["process_bathroom_daytime_stats"],
        domain=domain,
        module=module,
        reload=True,
        module_path=module_path,
        dependencies=parent_datasets["bathroom_daytime_fe"],
    )

    print("processing daily entropy FE")
    LocalDataset(
        dataset_name="entropy_daily_fe",
        datasets={d[0]: dcarte.load(*d) for d in parent_datasets["entropy_daily_fe"]},
        pipeline=["process_entropy_daily"],
        domain=domain,
        module=module,
        reload=True,
        module_path=module_path,
        dependencies=parent_datasets["entropy_daily_fe"],
    )

    print("processing bathroom transition FE")
    LocalDataset(
        dataset_name="bathroom_relative_transitions_fe",
        datasets={
            d[0]: dcarte.load(*d)
            for d in parent_datasets["bathroom_relative_transitions_fe"]
        },
        pipeline=["process_relative_transitions"],
        domain=domain,
        module=module,
        reload=True,
        module_path=module_path,
        dependencies=parent_datasets["bathroom_relative_transitions_fe"],
    )

    print("processing all FE")
    LocalDataset(
        dataset_name="all_fe",
        datasets={d[0]: dcarte.load(*d) for d in parent_datasets["all_fe"]},
        pipeline=["process_fe_data"],
        domain=domain,
        module=module,
        module_path=module_path,
        reload=True,
        dependencies=parent_datasets["all_fe"],
    )

    print("processing all core raw ")
    LocalDataset(
        dataset_name="core_raw",
        datasets={d[0]: dcarte.load(*d) for d in parent_datasets["core_raw"]},
        pipeline=["process_core_raw_data"],
        domain=domain,
        module=module,
        module_path=module_path,
        reload=True,
        dependencies=parent_datasets["core_raw"],
    )

    print("processing all core raw and FE")
    LocalDataset(
        dataset_name="all_core_raw_fe",
        datasets={d[0]: dcarte.load(*d) for d in parent_datasets["all_core_raw_fe"]},
        pipeline=["process_core_raw_and_fe_data"],
        domain=domain,
        module=module,
        module_path=module_path,
        reload=True,
        dependencies=parent_datasets["all_core_raw_fe"],
    )


if __name__ == "__main__":
    create_feature_engineering_datasets()
