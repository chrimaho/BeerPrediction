def all_str(lst):
    assert isinstance(lst, (list, str))
    return all([isinstance(element, str) for element in lst])

def all_bool(lst):
    assert isinstance(lst, (list, bool))
    return all([isinstance(element, bool) for element in lst])

def all_int(lst):
    assert isinstance(lst, (list, int))
    return all([isinstance(element, int) for element in lst])

def all_dataframe(lst):
    import pandas as pd
    assert isinstance(lst, (list, pd.DataFrame))
    return all([isinstance(element, pd.DataFrame) for element in lst])

def all_ndarray(lst):
    import numpy as np
    assert isinstance(lst, (list, np.ndarray))
    return all([isinstance(element, np.ndarray) for element in lst])

def all_dataframe_or_series(lst):
    import pandas as pd
    assert isinstance(lst, (list, pd.DataFrame, pd.Series))
    return all([isinstance(element, (pd.DataFrame, pd.Series)) for element in lst])