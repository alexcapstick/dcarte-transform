"""
Activity data functions.
"""

import pandas as pd
import numpy as np
import typing
import tqdm
import dcarte
from .utils import (
    compute_delta,
    lowercase_colnames,
    groupby_freq,
    between_time,
    collapse_levels,
)
from ..utils.progress import tqdm_style, pandarallel_progress

try:
    from pydtmc import MarkovChain

    pydtmc_import_error = False
except ImportError:
    pydtmc_import_error = True

try:
    from pandarallel import pandarallel as pandarallel_

    pandarallel_import_error = False
except ImportError:
    pandarallel_import_error = True


def compute_week_number(df: pd.DataFrame):
    """
    Compute the week number from the date.

    Arguments
    ---------

    - df:  pd.DataFrame:
        A data frame containing the dates to convert to week numbers.

    """
    df = pd.to_datetime(df, utc=True, infer_datetime_format=True)
    return df.dt.isocalendar().week + (df.dt.isocalendar().year - 2000) * 100


def compute_p_matrix(sequence, return_events=False):
    """
    This function allows the user to create a stochastic matrix from a
    sequence of events.


    Arguments
    ---------

    - sequence:  numpy.array:
        A sequence of events that will be used to calculate the stochastic matrix.

    - return_events:  bool, optional:
        Dictates whether a list of the events should be returned, in the
        order of their appearance in the stochastic matrix, :code:`p_martix`.
        Defaults to :code:`False`


    Returns
    --------

    - p_matrix:  numpy.array:
        A stochastic matrix, in which all of the rows sum to 1.

    - unique_locations:  list:
        A list of the events in the order of their appearance in the stochastic
        matrix, :code:`p_martix`. This is only returned if :code:`return_events=True`


    """

    # calculating transitions
    sequence_df = pd.DataFrame()
    sequence_df["from"] = sequence[:-1]
    sequence_df["to"] = sequence[1:]
    sequence_df["count"] = 1
    pm = sequence_df.groupby(by=["from", "to"]).count().reset_index()
    pm_total = pm.groupby(by="from")["count"].sum().to_dict()
    pm["total"] = pm["from"].map(pm_total)

    if pm.shape[0] < 2:
        return np.nan

    # calculating transition probabilities
    def calc_prob(x):
        return x["count"] / x["total"]

    pm["probability"] = pm.apply(calc_prob, axis=1)
    unique_locations = list(np.unique(pm[["from", "to"]].values.ravel()))
    p_matrix = np.zeros((len(unique_locations), len(unique_locations)))

    # calculating p matrix
    for (from_loc, to_loc, probability_loc) in pm[["from", "to", "probability"]].values:
        i = unique_locations.index(from_loc)
        j = unique_locations.index(to_loc)
        p_matrix[i, j] = probability_loc

    if return_events:
        return p_matrix, unique_locations
    else:
        return p_matrix


def compute_entropy_rate_from_sequence(sequence):
    """
    This function allows the user to calculate the entropy rate based on
    a sequence of events.



    Arguments
    ---------

    - sequence:  numpy.array:
        A sequence of events to calculate the entropy rate on.



    Returns
    --------

    - out:  float:
        Entropy rate


    """
    # imports required for parallel compute on windows
    import numpy as np

    try:
        from pydtmc import MarkovChain

        pydtmc_import_error = False
    except ImportError:
        pydtmc_import_error = True

    if pydtmc_import_error:
        raise ImportError(
            "pydtmc is required to calculate the entropy rate. "
            "Please install pydtmc>=6.10 to use this function."
        )

    p_matrix = compute_p_matrix(sequence)

    if type(p_matrix) != np.ndarray:
        return np.nan

    # we do not want to calculate the entropy for those graphs that
    # have a zero in the rows or only have a one in the rows,
    # since this is a consequence of cutting the sequences by a time period
    incomplete_rows = np.diag(p_matrix) == 1
    zero_rows = np.sum(p_matrix, axis=1) == 0
    if any(incomplete_rows) or any(zero_rows):
        return np.nan

    mc = MarkovChain(p_matrix)
    return mc.entropy_rate_normalized


@dcarte.utils.timer("calculating entropy rate")
def compute_entropy_rate(
    df: pd.DataFrame,
    id_col: str = "patient_id",
    datetime_col: str = "start_date",
    location_col: str = "location_name",
    sensors: typing.Union[typing.List[str], str] = "all",
    freq: typing.Union[typing.List[str], str] = ["day", "week"],
) -> typing.Union[pd.DataFrame, typing.List[pd.DataFrame]]:
    """
    This function allows the user to return a pandas.DataFrame with the entropy rate calculated
    for every week or day. The dataframe must contain :code:`[id_col]`, and columns containing the
    visited location names and the date and time of these location visits.


    Example
    ---------

    Note that the daily entropy will always be returned first in the list if two
    frequencies are given.

    .. code-block::

        >>> data = dcarte.load('activity','raw')
        >>> daily_entropy, weekly_entropy = compute_entropy_rate(data, freq=['day','week'])
        >>> daily_entropy, weekly_entropy = compute_entropy_rate(data, freq=['week','day'])




    Arguments
    ---------

    - df:  pandas.DataFrame:
        A data frame containing columns with the participant IDs,
        visited location names, and the date and time of these location visits.

    - id_col:  str, optional:
        The name of the column that contains the participant IDs.
        Defaults to :code:`'start_date'`.

    - datetime_col:  str, optional:
        The name of the column that contains the date time of location visits.
        Defaults to :code:`'start_date'`.

    - location_col:  str, optional:
        The name of the column that contains the location names visited.
        Defaults to :code:`'location_name'`.

    - sensors:  list` of :code:`str` or :code:`str:
        The values of the :code:`'location'` column of :code:`df` that will be
        used in the entropy calculations.
        Defaults to :code:`'all'`.

    - freq:  list` of :code:`str` or :code:`str:
        The period to calculate the entropy for. This can either be :code:`'day'`
        or :code:`'week'` or a list containing both.
        Defaults to :code:`['day', 'week']`



    Returns
    --------

    - out:  pd.DataFrame:
        returns a list of data frames containing the weekly and daily entropy
        or single dataframe if only one :code:`freq` was given.
        For weekly entropy, the date label corresponds to the start of the week,
        starting from Sunday.


    """

    from dcarte_transform.transform.activity import compute_entropy_rate_from_sequence

    if pandarallel_import_error:
        raise ImportError(
            "pandarallel is not installed, please install pandarallel>=1.6 to use this function."
        )

    assert len(sensors) >= 2, "need at least two sensors to calculate the entropy"

    df = df.sort_values(datetime_col).copy()

    if type(freq) == str:
        freq = [freq]

    # filter the sensors
    if isinstance(sensors, list):
        df = df[df.location.isin(sensors)]
    elif isinstance(sensors, str):
        assert sensors == "all", "Only accept 'all' as a string input for sensors"

    outputs = []

    # daily entropy calculations
    tqdm.tqdm.pandas(desc="Calculating daily entropy", **tqdm_style)
    if "day" in freq:
        # setting up parallel compute
        pandarallel_progress(desc="Computing daily entropy", smoothing=0, **tqdm_style)
        pandarallel_.initialize(progress_bar=True, verbose=0)

        daily_entropy = df[[id_col, datetime_col, location_col]].copy()
        daily_entropy = (
            daily_entropy.groupby(
                by=[id_col, pd.Grouper(key=datetime_col, freq="1d", label="left")]
            )
            .parallel_apply(
                lambda x: compute_entropy_rate_from_sequence(x[location_col].values)
            )
            .reset_index()
        )
        daily_entropy.columns = [id_col, "date", "daily_entropy"]
        daily_entropy["date"] = daily_entropy["date"].dt.date
        outputs.append(daily_entropy)

    # weekly entropy calculations
    tqdm.tqdm.pandas(desc="Calculating weekly entropy", **tqdm_style)
    if "week" in freq:

        pandarallel_progress(desc="Computing weekly entropy", smoothing=0, **tqdm_style)
        pandarallel_.initialize(progress_bar=True, verbose=0)

        weekly_entropy = df[[id_col, datetime_col, location_col]].copy()
        weekly_entropy = (
            weekly_entropy.groupby(
                by=[id_col, pd.Grouper(key=datetime_col, freq="W-SUN", label="left")]
            )
            .parallel_apply(
                lambda x: compute_entropy_rate_from_sequence(x[location_col].values)
            )
            .reset_index()
        )
        weekly_entropy.columns = [id_col, "date", "weekly_entropy"]
        weekly_entropy["date"] = weekly_entropy["date"].dt.date
        outputs.append(weekly_entropy)

    if len(outputs) > 1:
        return outputs
    else:
        return outputs[0]


def fill_from_first_occurence(df, subset=None, value=0):
    """
    This function fills each column in the subset
    in a dataframe with the value provided, from
    the first occurence

    This assumes that the dataframe is already
    sorted in the order in which you want to fill.


    """

    assert isinstance(df, pd.DataFrame), "df must be a pandas dataframe"

    df = df.copy()

    if subset is None:
        subset = df.columns
    elif type(subset) == str:
        subset = [subset]
    elif type(subset) != list:
        raise ValueError("subset must be a list of strings")

    for col in subset:
        first_index = df[col].first_valid_index()
        if first_index == None:
            continue
        df_temp = df.loc[first_index:, col]
        df_temp = df_temp.fillna(value)
        df.loc[first_index:, col] = df_temp

    return df


@dcarte.utils.timer("calculating daily location frequency")
def compute_daily_location_freq(
    df: pd.DataFrame,
    location: str,
    id_col: str = "patient_id",
    location_col: str = "location_name",
    datetime_col: str = "start_date",
    time_range: typing.Union[None, typing.List[str]] = None,
) -> pd.DataFrame:
    """
    This function allows you to calculate the frequency of visits to
    a given location during a given time range, aggregated daily.

    Example
    ---------

    To get the frequency of activity in the :code:`'bathroom1'` between
    the times of 00:00 to 08:00 and 20:00 to 00:00 each day, you could
    run the following:

    .. code-block::

        >>> compute_daily_location_freq(data, 'bathroom1', time_range=['20:00','08:00'])


    Arguments
    ---------

    - df:  pandas.DataFrame:
        The data frame containing the location visits to calculate the
        frequency from.

    - location:  str:
        The location name to calculate the frequencies for.

    - id_col:  str, optional:
        The name of the ID column that contains the participant IDs.
        Defaults to :code:`'patient_id'`.

    - location_col:  str, optional:
        The name of the location column that contains the visited location.
        Defaults to :code:`'location_name'`.

    - datetime_col: str, optional:
        The name of the location column that contains the date times
        of the location visits. This will be converted using
        :code:`pandas.to_datetime`.
        Defaults to :code:`'start_date'`.

    - time_range: None` or :code:`list` of :code:`str, optional:
        A time range given here, would allow you filter the frequencies
        by a given time. This allows you to calculate the frequencies
        of visits to a location during the night, for example. Acceptable
        arguments here are :code:`['[mm]:[ss]','[mm]:[ss]']`, in which
        the first element of the list is the start time and the second
        element is the end time.
        Defaults to :code:`None`.


    Returns
    ---------

    - table_of_frequencies:  pandas.DataFrame:
        The table containing the frequencies, with column names
        :code:`[id_col]`, :code:`'date'` and :code:`[name]` or
        :code:`[location]_freq`.



    """

    days_of_data = (
        df.astype({"location_name": object})
        .assign(date=lambda x: pd.to_datetime(x[datetime_col]).dt.date)[
            [id_col, "date"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    location_feq = (
        df[[id_col, datetime_col, location_col]]
        .astype({location_col: object, id_col: object})
        .assign(datetime_col=lambda x: pd.to_datetime(x[datetime_col]))
        .query(f"{location_col} == @location")
        .pipe(
            lambda df: pd.concat(
                [
                    df.pipe(between_time, time_range, datetime_col).assign(
                        time_range=True
                    ),
                    df.assign(time_range=False),
                ],
                axis=0,
            )
        )
        # counting the location counts for each day and each patient
        .pipe(
            groupby_freq,
            groupby_cols=[
                id_col,
                pd.Grouper(
                    key=datetime_col,
                    freq="1d",
                ),
                "time_range",
            ],
            count_col=location_col,
        )
        # unstacking to produce location name columns
        .unstack()
        # swapping the levels of the multiindex
        .swaplevel(i=0, j=1, axis=1)
        .pipe(collapse_levels)  # collapse multiindex columns
        .reset_index()
        .pipe(lowercase_colnames)  # lowercase column names
        # creating date column
        .assign(**{datetime_col: lambda x: pd.to_datetime(x[datetime_col]).dt.date})
        .rename(columns={datetime_col: "date"})
        # making 0 visited locations nan.
        # this is done because we want to fill the locations
        # with 0 from their first visit
        .pipe(
            lambda x: x.replace(
                {
                    col: {0: np.nan}
                    for col in list(
                        x.drop([id_col, "date", "time_range"], axis=1).columns
                    )
                }
            )
        )
        # filling from 0, including with the sensor
        # firings out of the time range. This ensures that even
        # if the sensor fired out of the time range,
        # we will still count the household as having that sensor
        .groupby(id_col, group_keys=False)
        .apply(
            lambda x: fill_from_first_occurence(
                x.sort_values(["date"]).sort_values(["time_range"], ascending=False),
                subset=list(x.drop([id_col, "date", "time_range"], axis=1).columns),
                value=0,
            )
        )
        .query("time_range != False")
        .drop("time_range", axis=1)
        # merging with all of the days of data, to ensure that all days are present
        # this is done because there might be days in which no locations in the core locations
        # were visited, but there were other locations visited
        .merge(days_of_data, on=[id_col, "date"], how="outer")
        .sort_values([id_col, "date"])
        .reset_index(drop=True)
        # filling the locations with 0 from their first visit
        .groupby(id_col, group_keys=False)
        .apply(
            lambda x: fill_from_first_occurence(
                x.sort_values("date"),
                subset=list(x.drop([id_col, "date"], axis=1).columns),
                value=0,
            )
        )
    )

    return location_feq


if __name__ == "__main__":
    data = dcarte.load("activity", "raw")
    entropy_daily, entropy_weekly = compute_entropy_rate(data, freq=["day", "week"])
    bathroom_feq = compute_daily_location_freq(data, location="bathroom1")
    bathroom_freq_daytime = compute_daily_location_freq(
        data, location="bathroom1", time_range=["08:00", "20:00"]
    )
    bathroom_freq_nighttime = compute_daily_location_freq(
        data, location="bathroom1", time_range=["20:00", "08:00"]
    )
    bathroom_freq_nighttime["bathroom1_freq_ma"] = (
        bathroom_freq_nighttime[["bathroom1_freq"]].rolling(3).mean()
    )
    bathroom_freq_nighttime["bathroom1_freq_ma_delta"] = compute_delta(
        bathroom_freq_nighttime["bathroom1_freq_ma"].values, pad=True
    )
