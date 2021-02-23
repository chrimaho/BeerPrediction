def dump_data(data:any, path:str="./data/raw", name:str="None", suffix:str="joblib"):
    """
    Dump the data from memory to a file.

    Args:
        data (any): The data to be dumped. Can be any type.
        path (str, optional): The directory where the data should be dumped to. Defaults to "./data/raw".
        name (str, optional): The name of the file that should be to. Defaults to "None".
        suffix (str, optional): The suffix of the file that will be dumped to. Defaults to "joblib".

    Raises:
        NotImplementedError: If the dumping fails for any reason.

    Returns:
        None: If successful, then nothing should be returned.
    """
    
    # Imports
    import os
    from joblib import dump
    
    # Assertions
    assert all([isinstance(param, str) for param in [path, name, suffix]])
    assert os.path.exists(path)
    
    # Join output name
    output = os.path.join(path, name) + "." + suffix
    
    # Do the dumping
    try:
        dump(data, output)
    except:
        raise NotImplementedError("Could not dump the data")
    
    return None