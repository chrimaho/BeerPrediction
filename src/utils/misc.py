
#------------------------------------------------------------------------------#
# get_*() funcitons                                                         ####
#------------------------------------------------------------------------------#

def get_name(object):
    # global env
    # env = globals()
    # return [name for name in env if env[name] is object]
    from varname import nameof,varname
    obj = varname(frame=2)
    return obj


def get_shape(object, return_str:bool=False):
    if not hasattr(object, "shape"):
        return "`object` has no shape."
    else:
        shape = object.shape
    if return_str:
        shape = " x ".join(str(dim) for dim in shape)
    return shape


def get_list_proportions(lst:list):
    """
    Get the proportions of each occurance of a class from within a list
    """
    import numpy as np
    assert isinstance(lst, (list, np.ndarray))
    prop = {}
    dist = list(set(lst))
    for val in dist:
        prop[val] = sum(map(lambda x: x==val, lst))/len(lst)
    return prop





#------------------------------------------------------------------------------#
# url functions                                                             ####
#------------------------------------------------------------------------------#

def valid_url(url):
    from urllib.parse import urlparse
    assert isinstance(url, str)
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False



#------------------------------------------------------------------------------#
# string manipulations                                                      ####
#------------------------------------------------------------------------------#


def str_right(string:str, num_chars:int):
    """
    Sub-Select the right-most number of characters from a string
    """
    assert isinstance(string, str)
    assert isinstance(num_chars, int)
    return string[-num_chars:]

