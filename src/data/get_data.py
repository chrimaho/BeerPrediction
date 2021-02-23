def get_file_data(url:str, save_path:str="./data/external", save_name:str="raw_data.tmp", chunk_size:int=128):
    """
    Import the data from a given URL, and save to a directory chunk by chunk.
    Inspiration for this function came from: https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url#answer-9419208

    Args:
        url (str): The URL from which the data will be downloaded.
        save_path (str, optional): The directory to which the data will be saved. Defaults to "./data/external".
        save_name (str, optional): The name of the file that will be saved. Defaults to "raw_data.tmp".
        chunk_size (int, optional): The chunk size of the data to be downloaded. Defaults to 128.

    Raises:
        ImportError: If there is an error with calling the API at any stage.
        Assertions: If any of the parameters are not the correct type or attribute.

    Returns:
        bool: Returns `True` if the Import+Export was successful
    """
    
    # Imports
    import os
    import requests
    from src.utils.misc import valid_url
    
    # Assertions
    assert all([isinstance(param, str) for param in [url, save_path, save_name]])
    assert all([isinstance(param, int) for param in [chunk_size]])
    assert valid_url(url)
    assert os.path.exists(save_path)
    
    # Get data
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        raise ImportError("Http Error: " + http_err)
    except requests.exceptions.ConnectionError as conn_err:
        raise ImportError("Connection Error: " + conn_err)
    except requests.exceptions.Timeout as time_err:
        raise ImportError("Timeout Error: " + time_err)
    except requests.exceptions.RequestException as excp_err:
        raise ImportError("Other Exception Error: " + excp_err)
    except Exception as err:
        raise ImportError("Unknown error occurred: " + err)       

    # Set file path
    target = os.path.join(save_path, save_name)
    
    # Save data
    with open(target, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
            
    return True
    
# Get the raw data ----
def get_raw_data(url:str):
    """
    Get raw data from the ABS website.
    For API formatting & error handling tips, see: https://realpython.com/python-requests/#status-codes
    
    Args:
        url (str): The url that should be called to get the raw data from.
        
    Raises:
        Assertions: If any of the parameters are not the correct type or attribute.
        ImportError: If the URL is invalid or if the API returns a bad status code.
        
    Returns:
        dict: The JSON output from the API response
    """
    
    # Imports
    import requests
    from src.utils.misc import valid_url
    
    # Assertions
    assert isinstance(url, str)
    assert valid_url(url)
    
    # Call the Api & handle the response
    try:
        response = requests.get(url, timeout=240)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        raise ImportError("Http Error: " + http_err)
    except requests.exceptions.ConnectionError as conn_err:
        raise ImportError("Connection Error: " + conn_err)
    except requests.exceptions.Timeout as time_err:
        raise ImportError("Timeout Error: " + time_err)
    except requests.exceptions.RequestException as excp_err:
        raise ImportError("Other Exception Error: " + excp_err)
    except Exception as err:
        raise ImportError("Unknown error occurred: " + err)
    else:
        call = response.json()
    
    # Return
    return call
