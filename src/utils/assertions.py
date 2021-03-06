#------------------------------------------------------------------------------#
# is_*() functions                                                           ####
#------------------------------------------------------------------------------#


def is_dataframe(obj):
    import pandas as pd
    return isinstance(obj, pd.DataFrame)


def is_series(obj):
    import pandas as pd
    return isinstance(obj, pd.Series)


def is_ndarray(obj):
    import numpy as np
    return isinstance(obj, np.ndarray)


def is_in(elem, lst):
    assert not has_len(elem)
    assert has_len(lst)
    return elem in lst


def is_valid_path(path:str):
    import os
    return os.path.exists(path)


def is_equal(obj1, obj2):
    return obj1==obj2



#------------------------------------------------------------------------------#
# has_*() funcions                                                          ####
#------------------------------------------------------------------------------#


def has_len(obj):
    return hasattr(obj, "__len__")



#------------------------------------------------------------------------------#
# all_*() functions                                                         ####
#------------------------------------------------------------------------------#


def all_in(sequence1, sequence2):
    """
    Confirm that all elements of one sequence are definitely contained within another
    """
    return all(elem in sequence2 for elem in sequence1)


def all_str(lst):
    if isinstance(lst, list):
        return all([isinstance(element, str) for element in lst])
    else:
        return isinstance(lst, str)
    

def all_bool(lst):
    if isinstance(lst, list):
        return all([isinstance(element, bool) for element in lst])
    else:
        return isinstance(lst, bool)


def all_int(lst):
    if isinstance(lst, list):
        return all([isinstance(element, int) for element in lst])
    else:
        return isinstance(lst, int)


def all_positive(lst):
    if isinstance(lst, list):
        return all([element > 0 for element in lst])
    else:
        return lst > 0


def all_dict(lst):
    if isinstance(lst, list):
        return all([isinstance(element, dict) for element in lst])
    else:
        return isinstance(lst, dict)

def all_real(lst):
    import numpy as np
    if isinstance(lst, list):
        return np.all([np.all(np.isreal(element)) for element in lst])
    else:
        return np.all(np.isreal(lst))


def all_scalar(lst):
    import numpy as np
    if isinstance(lst, list):
        return all([np.isscalar(element) for element in lst])
    else:
        return np.isscalar(lst)
    
    
def all_float(lst):
    if isinstance(lst, list):
        return all([isinstance(element, float) for element in lst])
    else:
        return isinstance(lst, float)


def all_float_or_int(lst):
    if isinstance(lst, list):
            return all([isinstance(element, (float,int)) for element in lst])
    else:
        return isinstance(lst, (float,int))


def all_dataframe(lst):
    if isinstance(lst, list):
        return all([is_dataframe(element) for element in lst])
    else:
        return is_dataframe(lst)


def all_ndarray(lst):
    if isinstance(lst, list):
        return all([is_ndarray(element) for element in lst])
    else:
        return is_ndarray(lst)


def all_dataframe_or_series(lst):
    import pandas as pd
    if isinstance(lst, list):
        return all([isinstance(element, (pd.DataFrame, pd.Series)) for element in lst])
    else:
        return isinstance(lst, (pd.DataFrame, pd.Series))

    
def all_dataframe_or_series_or_ndarray(lst):
    import pandas as pd
    import numpy as np
    if isinstance(lst, list):
        return all([isinstance(element, (pd.DataFrame, pd.Series, np.ndarray)) for element in lst])
    else:
        return isinstance(lst, (pd.DataFrame, pd.Series, np.ndarray))

    
def all_list(lst):
    return all([isinstance(element, list) for element in lst])


def all_valid_path(lst):
    from src.utils.assertions import is_valid_path
    if isinstance(lst, list):
        return all([is_valid_path(element) for element in lst])
    else:
        return is_valid_path(lst)
    
    
def all_equal(lst):
    return len(set(lst))==1