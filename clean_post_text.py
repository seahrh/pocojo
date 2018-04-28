import glob
import json
from os import path

import pandas as pd
from etl.markup_remover import strip_html
from stringx.stringx import to_ascii_str

posts_glob_pattern = 'posts/*.json'
file_path_template = 'posts_txt/p{}.txt'


def __title(jo):
    posts = jo['posts']
    if not posts:
        raise AssertionError('posts must not be empty')
    post = posts[0]
    res = to_ascii_str(post['title'])
    if not res:
        raise AssertionError('title must not be empty')
    return res


def __content(jo):
    posts = jo['posts']
    if not posts:
        raise AssertionError('posts must not be empty')
    post = posts[0]
    res = to_ascii_str(strip_html(post['content']))
    if not res:
        raise AssertionError('content must not be empty')
    return res


def __author(jo):
    posts = jo['posts']
    if not posts:
        raise AssertionError('posts must not be empty')
    post = posts[0]
    res = to_ascii_str(post['author']['display_name']).replace(' ', '').lower()
    if not res:
        raise AssertionError('author must not be empty')
    return res


def __save_text_file(file_path, text):
    with open(file_path, 'wt', encoding='utf8') as out:
        out.write(text)


def __pids(paths):
    res = []
    for p in paths:
        pid = path.basename(p)[1:-5]
        res.append(pid)
    return res


def __main():
    paths = glob.glob(posts_glob_pattern)
    pids = __pids(paths)
    df = pd.DataFrame(columns=['author', 'text'], index=pids)
    n = 0
    for p in paths:
        with open(p, 'rt') as f:
            pid = path.basename(p)[1:-5]
            n += 1
            print(f'n={n}, pid={pid}')
            jo = json.load(f)
            title = __title(jo)
            content = __content(jo)
            author = __author(jo)
            text = '{} {}'.format(title, content)
            df.loc[pid] = pd.Series({
                'author': author,
                'text': text
            })
            # file_path = file_path_template.format(pid)
            # __save_text_file(file_path, text)
    df.to_csv("data.tsv", sep='\t')


if __name__ == '__main__':
    __main()
