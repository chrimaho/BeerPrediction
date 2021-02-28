import pandas as pd

def make_ohe(feat:pd.DataFrame, cols:list, transform:bool=True):
    
    # Imports
    from src.utils import assertions as a
    from sklearn.preprocessing import OneHotEncoder
    
    # Assertions
    assert a.all_dataframe(feat)
    assert a.all_str(cols)
    if isinstance(cols, str): cols = [cols]
    assert a.all_in(cols, feat.columns)
    assert a.all_bool(transform)
    
    # Instantiations
    ohe = OneHotEncoder(sparse=False)
    
    # Do work
    data = feat[cols]
    if transform:
        data = ohe.fit_transform(data)
        data = pd.DataFrame(data)
        data.columns = ohe.get_feature_names(cols)
        feat.drop(cols, axis=1, inplace=True)
        feat = pd.concat([feat, data], axis=1)
    else:
        ohe.fit(data)
    
    # Return
    return feat, ohe
    
def make_oe(feat:pd.DataFrame, cols:list, transform:bool=True):
    
    # Imports
    from src.utils import assertions as a
    from sklearn.preprocessing import OrdinalEncoder
    
    # Assertions
    assert a.all_dataframe(feat)
    assert a.all_str(cols)
    if isinstance(cols, str): cols = [cols]
    assert a.all_in(cols, feat.columns)
    assert a.all_bool(transform)
    
    # Instantiations
    oe = OrdinalEncoder()
    
    # Do work
    if transform:
        feat[cols] = oe.fit_transform(feat[cols])
    else:
        oe.fit(feat[cols])
    
    # Return
    return feat, oe


def make_le(feat:pd.DataFrame, cols:list, transform:bool=True):
    
    # Imports
    from src.utils import assertions as a
    from sklearn.preprocessing import LabelEncoder
    
    # Assertions
    assert a.all_dataframe(feat)
    assert a.all_str(cols)
    if isinstance(cols, str): cols = [cols]
    assert a.all_in(cols, feat.columns)
    assert a.all_bool(transform)
    
    # Instantiations
    le = LabelEncoder()
    
    # Do work
    if transform:
        feat[cols] = le.fit_transform(feat[cols])
    else:
        le.fit(feat[cols])
    
    # Return
    return feat, le


def make_si(feat:pd.DataFrame, cols:list, transform:bool=True):
    
    # Imports
    from src.utils import assertions as a
    from sklearn.impute import SimpleImputer
    import numpy as np
    
    # Assertions
    assert a.all_dataframe(feat)
    if isinstance(cols, str): cols = [cols]
    assert a.all_str(cols)
    assert a.all_in(cols, feat.columns)
    assert a.all_bool(transform)
    
    # Instantiations
    si = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="Other")
    
    # Do work
    if transform:
        feat[cols] = si.fit_transform(feat[cols])
    else:
        si.fit(feat[cols])
    
    # Return
    return feat, si
    


def encode_features(feat:pd.DataFrame, cols=list, type:str="ordinal", transform:bool=True):
    
    # Imports
    from src.utils import assertions as a
    from src.data.prep_data import make_oe, make_ohe
    
    # Assertions
    assert a.all_dataframe(feat)
    assert isinstance(cols, (str, list))
    assert a.all_str(cols)
    assert a.all_str(type)
    
    # Do work & return
    if type in ["oe","ord","ordinal","ordinalencoder","ordinal encoder"]:
        return make_oe(feat=feat, cols=cols, transform=transform)
    elif type in ["ohe","one","onehotencoder","one hot encoder"]:
        return make_ohe(feat=feat, cols=cols, transform=transform)
    elif type in ["le","label","label encoder","labelencoder"]:
        return make_le(feat=feat, cols=cols, transform=transform)
    else:
        return feat, None
    
def scale_features(feat:pd.DataFrame, cols:list=None, transform:bool=True):
    
    # Imports
    from src.utils import assertions as a
    from sklearn.preprocessing import StandardScaler
    
    # Assertions
    assert a.all_dataframe(feat)
    if cols:
        if isinstance(cols, str): cols = [cols]
        assert a.all_str(cols)
        assert a.all_in(cols, feat.columns)
    assert a.all_bool(transform)
    
    # Get cols
    if not cols:
        cols = feat.columns
    
    # Instantiations
    sc = StandardScaler()
    
    # Do work
    if transform:
        feat[cols] = sc.fit_transform(feat[cols])
    else:
        sc.fit(feat[cols])
    
    # Return
    return feat, sc