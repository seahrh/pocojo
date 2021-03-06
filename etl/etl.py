from urllib import request
from urllib.error import HTTPError
from sgcharts.stringx import to_str


def scrape_to_file(url,
                   headers,
                   file_path,
                   timeout=30,
                   file_encoding='utf8'):
    req = request.Request(url, headers=headers)
    try:
        u = request.urlopen(req, timeout=timeout)
    except HTTPError as e:
        if e.code == 404:
            print(f'404: {url}')
        else:
            raise
    else:
        print(f'200: {url}')
        s = to_str(u.read())
        # print(f's={s}')
        with open(file_path, 'wt', encoding=file_encoding) as f:
            f.write(s)
        return True
    return False
