from urllib import request, parse
from urllib.error import HTTPError
from time import sleep
from random import uniform
import json

base_url = 'https://www.techinasia.com/wp-json/techinasia/2.0/posts/'
pid_min = 491009
pid_max = 491011
sleep_sec_min = 0.3
sleep_sec_max = 1
headers = {
    'User-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
}


def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value # Instance of str


def sleep_jitter(min, max):
    sleep_sec = uniform(min, max)
    # print(f'sleep_sec={sleep_sec}')
    sleep(sleep_sec)


for i in range(pid_min, pid_max + 1):
    url = base_url + str(i)
    print(f'url={url}')
    req = request.Request(url, headers=headers)
    try:
        u = request.urlopen(req, timeout=30)
    except HTTPError as e:
        if e.code == 404:
            print(f'pid={i}, 404 Not Found')
        else:
            raise
    else:
        json_str = to_str(u.read())
        p = json.loads(json_str)
        print(f'p={p}')
        # print(f'json_str={json_str}')
        with open(f'posts/p{i}.json', 'wt', encoding='utf8') as f:
            f.write(json_str)
    sleep_jitter(sleep_sec_min, sleep_sec_max)





