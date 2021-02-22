def download_data(url:str, save_path:str="./data/raw", save_name:str="raw_data.tmp", chunk_size:int=128):
    # https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url#answer-9419208
    
    # Imports
    import os
    import requests
    
    # Assertions
    assert all([isinstance(param, str) for param in [url, save_path, save_name]])
    assert all([isinstance(param, int) for param in [chunk_size]])
    assert os.path.exists(save_path)
    
    # Get data
    r = requests.get(url, stream=True)
    
    # Save data
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    
    
    
# Get the raw data ----
def get_RawData(url:str):
    """
    Get raw data from the ABS website.
    For API formatting & error handling tips, see: https://realpython.com/python-requests/#status-codes
    
    Args:
        url (str): The url that should be called to get the raw data from.
        
    Raises:
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
