from transform.activity import fill_from_first_occurence

import numpy as np
import pandas as pd

def test_fill_from_first_occurence():
    
    # testing whether it fills correctly
    df = pd.DataFrame(
        {'0':[1,2,3,4,5], '1':[1,2,3,4,5], '2':[1,2,3,4,5]}
    )
    df.loc[2, '0'] = np.nan
    df.loc[3, '1'] = np.nan
    df.loc[4, '2'] = np.nan

    assert \
        fill_from_first_occurence(
            df, 
            subset=['0', '1', '2'],
            value=0
        ).equals(
            pd.DataFrame(
            {'0':[1,2,0,4,5], '1':[1,2,3,0,5], '2':[1,2,3,4,0]}
        ).astype(float)
    )

    # making sure it doesn't fill in this instance, 
    # since nan is the first occurence and a whole column is nan
    df = pd.DataFrame(
        {'0':[1,2,3,4,5], '1':[1,2,3,4,5], '2':[1,2,3,4,5]}
    )
    df.loc[0, '0'] = np.nan
    df.loc[:, '1'] = np.nan

    assert \
        fill_from_first_occurence(
            df, 
            subset=['0', '1', '2'],
            value=0
        ).equals(df)

    # testing whether it fills when no subset is given
    df = pd.DataFrame(
        {'0':[1,2,3,4,5], '1':[1,2,3,4,5], '2':[1,2,3,4,5]}
    )
    df.loc[2, '0'] = np.nan
    df.loc[3, '1'] = np.nan
    df.loc[4, '2'] = np.nan

    assert \
        fill_from_first_occurence(
            df, 
            value=0
        ).equals(
            pd.DataFrame(
            {'0':[1,2,0,4,5], '1':[1,2,3,0,5], '2':[1,2,3,4,0]}
        ).astype(float)
    )
