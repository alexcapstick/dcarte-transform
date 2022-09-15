'''
Labelling all data. This provides the most useful functionality.
'''

import typing
import pandas as pd

from .agitation import label as _label_agitation
from .agitation import get_labels as _get_labels_agitation
from .agitation import label_number_previous as _label_number_previous_agitation

from .uti import label as _label_uti
from .uti import get_labels as _get_labels_uti
from .uti import label_number_previous as _label_number_previous_uti




class Labeller:
    def __init__(
        self,
        ):
        '''
        This function allows the user to label data.

        The labelling types available can be accessed through the attribute
        `.label_types`.
        

        Examples
        ---------
        ```
        >>> l = Labeller()
        >>> l.label_types
        ['uti', 'agitation']
        >>> all_labels = l.get_labels(
            days_either_side=2,
            return_event=True,
            )
        >>> data_labelled = l.label_df(data, subset='uti') # this is the same as below
        >>> data_labelled = l.uti_label_df(data) # this is the same as above
        ```

        '''
        self.label_types = ['uti', 'agitation']

        return
    
    def agitation_label_df(
        self,
        df:pd.DataFrame, 
        id_col:str='patient_id', 
        datetime_col:str='start_date', 
        days_either_side:int=0, 
        return_event:bool=False,
        ) -> pd.DataFrame:
        '''
        This method will label the input dataframe based on the agitation data 
        in `behaviour`.

        Arguments
        ----------
        
        - `df`: `pandas.DataFrame`:
            Unlabelled dataframe, must contain columns `[id_col, datetime_col]`, where `id_col` is the
            ids of participants and `datetime_col` is the time of the sensors.

        - `id_col`: `str`, optional:
            The column name that contains the ID information.
            Defaults to `'patient_id'`.

        - `datetime_col`: `str`, optional:
            The column name that contains the date time information.
            Defaults to `'start_date'`.

        - `days_either_side`: `int`, optional:
            The number of days either side of a label that will be given the same label.
            Defaults to `0`.
        
        - `return_event`: `bool`, optional:
            This dictates whether another column should be added, with a unique id given to each of the separate
            agitation events. This allows the user to group the outputted data based on events.
            Defaults to `False`.

        Returns
        ---------
        
        - df_labelled: `pandas.DataFrame`:
            This is a dataframe containing the original data along with a new column, `'agitation_labels'`,
            which contains the labels. If `return_event=True`, a column titled `'agitation_event'` will be 
            added which contains unique IDs for each of the agitation episodes.
        '''

        return _label_agitation(
            df=df,
            id_col=id_col,
            datetime_col=datetime_col,
            days_either_side=days_either_side,
            return_event=return_event,
            )


    def uti_label_df(
        self,
        df:pd.DataFrame, 
        id_col:str='patient_id', 
        datetime_col:str='start_date', 
        days_either_side:int=0, 
        return_event:bool=False,
        ) -> pd.DataFrame:
        '''
        This method will label the input dataframe based on the uti data 
        in `procedure`.

        Arguments
        ----------
        
        - `df`: `pandas.DataFrame`:
            Unlabelled dataframe, must contain columns `[id_col, datetime_col]`, where `id_col` is the
            ids of participants and `datetime_col` is the time of the sensors.

        - `id_col`: `str`, optional:
            The column name that contains the ID information.
            Defaults to `'patient_id'`.

        - `datetime_col`: `str`, optional:
            The column name that contains the date time information.
            Defaults to `'start_date'`.

        - `days_either_side`: `int`, optional:
            The number of days either side of a label that will be given the same label.
            Defaults to `0`.
        
        - `return_event`: `bool`, optional:
            This dictates whether another column should be added, with a unique id given to each of the separate
            UTI events. This allows the user to group the outputted data based on events.
            Defaults to `False`.

        Returns
        ---------
        
        - df_labelled: `pandas.DataFrame`:
            This is a dataframe containing the original data along with a new column, `'uti_labels'`,
            which contains the labels. If `return_event=True`, a column titled `'uti_event'` will be 
            added which contains unique IDs for each of the UTI episodes.
        '''

        return _label_uti(
            df=df,
            id_col=id_col,
            datetime_col=datetime_col,
            days_either_side=days_either_side,
            return_event=return_event,
            )


    def get_agitation_labels(
        self,
        days_either_side:int=0, 
        return_event:bool=False,
        ) -> pd.DataFrame:
        '''
            This method will return the Agitation labels.
            If a single day for a paticular ID contains two different
            labels (usually caused by using `days_either_side`),
            then both labels are removed.

            Arguments
            ---------

            - `days_either_side`: `int`, optional:
                The number of days either side of a label that will be given the same label.
                If these days overlap, if the label is the same then the first will be kept.
                If they are different, then neither will be kept.
                Defaults to `0`.

            - `return_event`: `bool`, optional:
                This dictates whether another column should be added, with a unique id given to each of the separate
                UTI events. This allows the user to group the outputted data based on events.
                Defaults to `False`.


            Returns
            --------

            - `out`: `pd.DataFrame` :
                A dataframe containing the Agitation labels, with the corresponding patient_id and
                date.

            '''

        df_out = _get_labels_agitation(
            days_either_side=days_either_side, 
            return_event=return_event,
            )
        
        return df_out.rename(columns={'outcome': 'agitation_label', 'event': 'agitation_event'})


    def get_uti_labels(
        self,
        days_either_side:int=0, 
        return_event:bool=False,
        ) -> pd.DataFrame:
        '''
        This method will return the UTI labels.
        If a single day for a paticular ID contains two different
        labels (usually caused by using `days_either_side`),
        then both labels are removed.
        
        
        
        Arguments
        ---------
        
        - `days_either_side`: `int`, optional:
            The number of days either side of a label that will be given the same label.
            If these days overlap, if the label is the same then the first will be kept.
            If they are different, then neither will be kept.
            Defaults to `0`.
        
        - `return_event`: `bool`, optional:
            This dictates whether another column should be added, with a unique id given to each of the separate
            UTI events. This allows the user to group the outputted data based on events.
            Defaults to `False`.
        
        
        Returns
        --------
        
        - `out`: `pd.DataFrame` : 
            A dataframe containing the uti labels, with the corresponding patient_id and 
            date.

        '''

        df_out = _get_labels_uti(
            days_either_side=days_either_side, 
            return_event=return_event,
            )
        
        return df_out.rename(columns={'outcome': 'uti_label', 'event': 'uti_event'})
    

    def previous_agitation_label_df(
        self,
        df:pd.DataFrame, 
        id_col:str='patient_id', 
        datetime_col:str='start_date',
        day_delay:int=1,
        ):
        '''
        This method allows you to label the number of agitation positives to date
        for the corresponding ID and date.
        
        Arguments
        ---------
        - `df`: `pandas.DataFrame`:
            The dataframe to append the number of previous agitation positives to.
        
        - `id_col`: `str`, optional:
            The column name that contains the ID information.
            Defaults to `'patient_id'`.

        - `datetime_col`: `str`, optional:
            The column name that contains the date time information.
            Defaults to `'start_date'`.

        - `day_delay`: `str`, optional:
            The number of days after an agitation is detected when the data reflects
            that the ID has had another previous agitation. This is used to ensure
            that the predictive model does not simply learn that to look for 
            when this feature increases.
            Defaults to `1`.
        
        Returns
        ---------
        
        - df_out: `pandas.DataFrame`:
            This is a dataframe containing the original data along with a new column, `'agitation_previous'`,
            which contains the number of previous agitations to date for that ID.
        
        
        '''

        return _label_number_previous_agitation(
            df=df,
            id_col=id_col,
            datetime_col=datetime_col,
            day_delay=day_delay,
            )
    

    def previous_uti_label_df(
        self,
        df:pd.DataFrame, 
        id_col:str='patient_id', 
        datetime_col:str='start_date',
        day_delay:int=1,
        ):
        '''
        This function allows you to label the number of uti positives to date
        for the corresponding ID and date.
        
        Arguments
        ---------
        - `df`: `pandas.DataFrame`:
            The dataframe to append the number of previous uti positives to.
        
        - `id_col`: `str`, optional:
            The column name that contains the ID information.
            Defaults to `'patient_id'`.

        - `datetime_col`: `str`, optional:
            The column name that contains the date time information.
            Defaults to `'start_date'`.

        - `day_delay`: `str`, optional:
            The number of days after a UTI is detected when the data reflects
            that the ID has had another previous UTI. This is used to ensure
            that the predictive model does not simply learn that to look for 
            when this feature increases.
            Defaults to `1`.
        
        Returns
        ---------
        
        - df_out: `pandas.DataFrame`:
            This is a dataframe containing the original data along with a new column, `'uti_previous'`,
            which contains the number of previous UTIs to date for that ID.
        
        
        '''

        return _label_number_previous_uti(
            df=df,
            id_col=id_col,
            datetime_col=datetime_col,
            day_delay=day_delay,
            )
    

    def label_df(
        self,
        df:pd.DataFrame, 
        subset:typing.Union[None, str, typing.List[str]]=None,
        id_col:str='patient_id', 
        datetime_col:str='start_date', 
        days_either_side:int=0, 
        return_event:bool=False,
        ) -> pd.DataFrame:
        '''
        This method will label the input dataframe based on the subset
        of labels given.

        Arguments
        ----------
        
        - `df`: `pandas.DataFrame`:
            Unlabelled dataframe, must contain columns `[id_col, datetime_col]`, where `id_col` is the
            ids of participants and `datetime_col` is the time of the sensors.
        
        - `subset`: `typing.Union[None, str, typing.List[str]]`:
            The subset of label types to be used in the labelling.
            If `None`, then all label types will be used. These can 
            be accessed using the attribute `.label_types`.
            Defaults to `None`.

        - `id_col`: `str`, optional:
            The column name that contains the ID information.
            Defaults to `'patient_id'`.

        - `datetime_col`: `str`, optional:
            The column name that contains the date time information.
            Defaults to `'start_date'`.

        - `days_either_side`: `int`, optional:
            The number of days either side of a label that will be given the same label.
            Defaults to `0`.
        
        - `return_event`: `bool`, optional:
            This dictates whether another column should be added, with a unique id given to each of the separate
            label events. This allows the user to group the outputted data based on events.
            Defaults to `False`.

        Returns
        ---------
        
        - df_labelled: `pandas.DataFrame`:
            This is a dataframe containing the original data along with new columns
            which contains the labels. If `return_event=True`, a columns will be 
            added which contains unique IDs for each of the label episodes.
        '''
        
        if subset is None:
            subset = self.label_types
        elif type(subset) == str:
            subset = [subset]
        elif type(subset) == list:
            pass
        else:
            raise TypeError("Please ensure that subset is a str, list of str or None.")

        for label_type in subset:
            df = getattr(self, f'{label_type}_label_df')(
                df=df,
                id_col=id_col,
                datetime_col=datetime_col,
                days_either_side=days_either_side,
                return_event=return_event,
                )
        
        return df


    def previous_label_df(
        self,
        df:pd.DataFrame, 
        subset:typing.Union[None, str, typing.List[str]]=None,
        id_col:str='patient_id', 
        datetime_col:str='start_date',
        day_delay:int=1,
        ) -> pd.DataFrame:
        '''
        This function allows you to label the number of positives to date
        for the corresponding ID and date from the subset given.
        
        Arguments
        ---------
        - `df`: `pandas.DataFrame`:
            The dataframe to append the number of previous positives to.
        
        - `subset`: `typing.Union[None, str, typing.List[str]]`:
            The subset of label types to be used in the labelling.
            If `None`, then all label types will be used. These can 
            be accessed using the attribute `.label_types`.
            Defaults to `None`.
        
        - `id_col`: `str`, optional:
            The column name that contains the ID information.
            Defaults to `'patient_id'`.

        - `datetime_col`: `str`, optional:
            The column name that contains the date time information.
            Defaults to `'start_date'`.

        - `day_delay`: `str`, optional:
            The number of days after a label is detected when the data reflects
            that the ID has had another previous label. This is used to ensure
            that the predictive model does not simply learn that to look for 
            when this feature increases.
            Defaults to `1`.
        
        Returns
        ---------
        
        - df_out: `pandas.DataFrame`:
            This is a dataframe containing the original data along with new columns
            which contains the number of previous UTIs to date for that ID.
        
        
        '''
        
        if subset is None:
            subset = self.label_types
        elif type(subset) == str:
            subset = [subset]
        elif type(subset) == list:
            pass
        else:
            raise TypeError("Please ensure that subset is a str, list of str or None.")

        for label_type in subset:
            df = getattr(self, f'previous_{label_type}_label_df')(
                df=df,
                id_col=id_col,
                datetime_col=datetime_col,
                day_delay=day_delay,
                )
        
        return df


    def get_labels(
        self,
        subset:typing.Union[None, str, typing.List[str]]=None,
        days_either_side:int=0, 
        return_event:bool=False,
        ) -> pd.DataFrame:
        '''
        This method will return the labels from the subset given.
        If a single day for a paticular ID contains two different
        labels (usually caused by using `days_either_side`),
        then both labels are removed.
        
        
        
        Arguments
        ---------
        
        - `subset`: `typing.Union[None, str, typing.List[str]]`:
            The subset of label types to be used in the labelling.
            If `None`, then all label types will be used. These can 
            be accessed using the attribute `.label_types`.
            Defaults to `None`.
        
        - `days_either_side`: `int`, optional:
            The number of days either side of a label that will be given the same label.
            If these days overlap, if the label is the same then the first will be kept.
            If they are different, then neither will be kept.
            Defaults to `0`.
        
        - `return_event`: `bool`, optional:
            This dictates whether another column should be added, with a unique id given to each of the separate
            label events. This allows the user to group the outputted data based on events.
            Defaults to `False`.
        
        
        Returns
        --------
        
        - `out`: `pd.DataFrame` : 
            A dataframe containing the labels, with the corresponding patient_id and 
            date.

        '''

        if subset is None:
            subset = self.label_types
        elif type(subset) == str:
            subset = [subset]
        elif type(subset) == list:
            pass
        else:
            raise TypeError("Please ensure that subset is a str, list of str or None.")
        
        labels = pd.DataFrame({'patient_id': [], 'date': []})
        for label_type in subset:
            label_temp = getattr(self, f'get_{label_type}_labels')(
                days_either_side=days_either_side, 
                return_event=return_event,
                )

            labels = pd.merge(
                left=labels, 
                right=label_temp,
                how='outer',
                on=['patient_id', 'date'],
                )
        
        return labels
