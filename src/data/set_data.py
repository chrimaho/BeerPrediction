import pandas as pd

def rem_features(data:pd.DataFrame, feats:list):
    
    # Imports
    from src.utils import assertions as a
    
    # Assertions
    assert a.all_dataframe(data)
    assert a.all_str(feats, (str, list))
    assert a.all_str(feats)
    assert a.all_in(feats, data.columns)
    
    # Do work
    data = data.drop(columns=feats)
    
    # Return
    return data

def sel_feat_cols(data:pd.DataFrame, feats:list):
    
    # Imports
    from src.utils import assertions as a
    
    # Assertions
    assert a.all_dataframe(data)
    assert isinstance(feats, (str, list))
    assert a.all_str(feats)
    assert a.all_in(feats, data.columns)
    
    # Do work
    data = data[feats]
    
    return data

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
    from src.utils import assertions as a
    
    # Assertions
    assert a.all_dataframe(data)
    assert a.all_str(targ)
    assert targ in data.columns
    
    # Do work
    # targ = data.pop(targ)
    feat = data.drop([targ], axis=1)
    targ = data[[targ]]
    
    # Return
    return feat, targ


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
    from src.utils import assertions as a
    from sklearn.model_selection import train_test_split
    
    # Assertions
    assert a.all_dataframe_or_series([feat, targ])
    assert not all([train_size is None, test_size is None]), "Must provide value for at least one of `train_size` or `test_size`."
    if train_size is not None and test_size is not None:
        assert all([param>0 and param<1 for param in [train_size, test_size]])
        assert test_size+train_size==1, "If providing a value for both, the sum of `train_size` and `test_size` must equal `1`."
    assert a.all_int([random_state])
    assert random_state>0
    
    # Do work
    if test_size:
        feat_trn, feat_tst, targ_trn, targ_tst = train_test_split(feat, targ, stratify=targ, test_size=test_size, random_state=random_state)
    elif train_size:
        feat_trn, feat_tst, targ_trn, targ_tst = train_test_split(feat, targ, stratify=targ, train_size=train_size, random_state=random_state)

    # Return
    return feat_trn, feat_tst, targ_trn, targ_tst


def split_sets_random(df, target_col, test_ratio=0.2, to_numpy=False):
    """Split sets randomly

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    
    from sklearn.model_selection import train_test_split
    
    features, target = pop_target(df=df, target_col=target_col, to_numpy=to_numpy)
    
    X_data, X_test, y_data, y_test = train_test_split(features, target, test_size=test_ratio, random_state=8)
    
    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=val_ratio, random_state=8)

    return X_train, y_train, X_val, y_val, X_test, y_test