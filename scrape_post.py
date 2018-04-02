from random import uniform
from time import sleep

from etl import etl

url_template = 'https://www.techinasia.com/wp-json/techinasia/2.0/posts/{}'
pid_min = 299000
pid_max = 299999
sleep_sec_min = 0.3
sleep_sec_max = 1
file_path_template = 'posts/p{}.json'
headers = {
    'User-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
}


def sleep_jitter(min_sec, max_sec):
    sec = uniform(min_sec, max_sec)
    # print(f'sleep_sec={sleep_sec}')
    sleep(sec)


success_n = 0
for i in range(pid_min, pid_max + 1):
    url = url_template.format(i)
    file_path = file_path_template.format(i)
    res = etl.scrape_to_file(url, headers, file_path)
    if res:
        success_n += 1
        print(f'success_n={repr(success_n)}')
    sleep_jitter(sleep_sec_min, sleep_sec_max)
