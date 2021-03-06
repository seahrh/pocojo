from time import sleep
from random import uniform
import glob
from os import path
from etl import etl

url_template = 'https://www.techinasia.com/wp-json/techinasia/2.0/posts/{}/comments'
sleep_sec_min = 0.3
sleep_sec_max = 1
pid_min = 10000
pid_max = 299013
posts_glob_pattern = 'posts/*.json'
file_path_template = 'comments/c{}.json'
headers = {
    'User-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
}


def sleep_jitter(min_sec, max_sec):
    sec = uniform(min_sec, max_sec)
    # print(f'sleep_sec={sleep_sec}')
    sleep(sec)


post_files = glob.glob(posts_glob_pattern)
success_n = 0
for p in post_files:
    pid = path.basename(p)[1:-5]
    # print(f'pid={repr(pid)}')
    if pid_min <= int(pid) <= pid_max:
        url = url_template.format(pid)
        file_path = file_path_template.format(pid)
        res = etl.scrape_to_file(url, headers, file_path)
        if res:
            success_n += 1
            print(f'success_n={repr(success_n)}')
        sleep_jitter(sleep_sec_min, sleep_sec_max)
