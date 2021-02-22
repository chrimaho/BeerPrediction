def valid_url(url):
    from urllib.parse import urlparse
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
