def valid_url(url):
    from urllib.parse import urlparse
    assert isinstance(url, str)
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

def get_name(object):
    return [name for name in globals() if globals()[name] is object]