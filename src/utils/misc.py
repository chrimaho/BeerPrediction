def valid_url(url):
    from urllib.parse import urlparse
    assert isinstance(url, str)
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

def get_name(object, env=globals()):
    return [name for name in env if env[name] is object]

def get_shape(object, return_str:bool=False):
    shape = object.shape
    if return_str:
        shape = " ".join(str(dim) for dim in shape)
    return shape

def all_in(sequence1, sequence2):
    """
    Confirm that all elements of one sequence are definitely contained within another
    """
    return all(elem in sequence2 for elem in sequence1)

def str_right(string:str, num_chars:int):
    """
    Sub-Select the right-most number of characters from a string
    """
    assert isinstance(string, str)
    assert isinstance(num_chars, int)
    return string[-num_chars:]

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