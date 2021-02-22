def download_data(url:str, save_path:str="./data/raw", save_name:str="raw_data.tmp", chunk_size:int=128):
    # https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url#answer-9419208
    
    # Imports
    import requests
    import os
    
    # Assertions
    assert all([isinstance(param, str) for param in [url, save_path, save_name]])
    assert all([isinstance(param, int) for param in [chunk_size]])
    
    # Get data
    r = requests.get(url, stream=True)
    
    # Save data
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    
    