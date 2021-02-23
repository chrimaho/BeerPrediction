import pandas as pd

def pop_target(data:pd.DataFrame, targ:str):
    """
    Pop the target column off the data set.

    Args:
        data (pd.DataFrame): The data set, from which the target will be removed.
        targ (str): The name of the feature to be removed. Must be a valid and existing column in `data`.

    Returns:
        data (pd.DataFrame): The updated `data` object, having had the `feat` column removed.
        targ (pd.Series): The feature that has been removed from the `data` object.
    """
    
    # Imports
    import pandas as pd
    
    # Assertions
    assert isinstance(data, pd.DataFrame)
    assert targ in data.columns
    
    # Do work
    targ = data.pop(targ)
    
    # Return
    return data, targ

def split_data(feat:pd.DataFrame, targ:pd.DataFrame, train_size:float=None, test_size:float=None, random_state:int=123):
    """
    Split the data in to train & test sets.
    Note:
        1. The length of `feat` and `targ` must be the same.
        2. You can parse one or both of the `train_size` or `test_size` parameters.
        3. If you parse one, the other assumed as: 1 minus the value of the param given.
        4. If you parse both, the sum of the two must equal 1.

    Args:
        feat (pd.DataFrame): The feature data set.
        targ (pd.DataFrame): The target data set.
        train_size (float, optional): The percentage (as a decimal) of the data to be extracted as the training size . Defaults to None.
        test_size (float, optional): [description]. Defaults to None.
        random_state (int, optional): [description]. Defaults to 123.
    """
    
    # Imports
    from sklearn.model_selection import train_test_split
    
    # Assertions
    assert all([isinstance(param, (pd.DataFrame, pd.Series)) for param in [feat, targ]])
    assert all([isinstance(param, type(None)) for param in [train_size, test_size]])
    assert train_size is None and test_size is None, "Must provide value for at least one of `train_size` or `test_size`."
    if train_size is not None and test_size is not None:
        assert all([param>0 and param<1 for param in [train_size, test_size]])
        assert test_size+train_size==1, "If providing a value for both, the sum of `train_size` and `test_size` must equal `1`."
    assert isinstance(random_state, int)
    assert random_state>0
    
    # Do work
    if test_size:
        feat_trn, feat_tst, targ_trn, targ_tst = train_test_split(feat, targ, stratify=targ, test_size=test_size, random_state=random_state)
    elif train_size:
        feat_trn, feat_tst, targ_trn, targ_tst = train_test_split(feat, targ, stratify=targ, train_size=train_size, random_state=random_state)

    # Return
    feat_trn, feat_tst, targ_trn, targ_tst