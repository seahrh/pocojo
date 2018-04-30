import glob
import json
from os import path

import pandas as pd
from etl.markup_remover import strip_html
from stringx.stringx import to_ascii_str

__posts_glob_pattern = 'posts/*.json'
__file_path_template = 'posts_txt/p{}.txt'
__comment_path_template = 'comments/c{}.json'
__out_file_path = 'tmp/data.tsv'


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


def __comment_count(file_path):
    with open(file_path, 'rt') as f:
        jo = json.load(f)
        comments = jo['comments']
        n = 0
        for i, comment in enumerate(comments):
            n += 1
            for _ in comment['children']:
                n += 1
        return n


def __main():
    paths = sorted(glob.glob(__posts_glob_pattern))
    pids = __pids(paths)
    df = pd.DataFrame(columns=['comment_count', 'author', 'text'], index=pids)
    n = 0
    for p in paths:
        with open(p, 'rt') as f:
            pid = path.basename(p)[1:-5]
            n += 1
            comment_path = __comment_path_template.format(pid)
            comment_count = __comment_count(comment_path)
            print(f'n={n}, pid={pid}, comment_count={comment_count}')
            jo = json.load(f)
            title = __title(jo)
            content = __content(jo)
            author = __author(jo)
            text = '{} {}'.format(title, content)
            df.loc[pid] = pd.Series({
                'comment_count': comment_count,
                'author': author,
                'text': text
            })
            # file_path = file_path_template.format(pid)
            # __save_text_file(file_path, text)
    df = pd.get_dummies(df, columns=['author'], prefix=['author'])
    df.to_csv(__out_file_path, sep='\t')


if __name__ == '__main__':
    __main()
