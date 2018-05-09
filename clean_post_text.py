import glob
import json
from os import path
from pprint import pprint

import numpy as np
import pandas as pd
from etl.markup_remover import strip_html
from stringx.stringx import to_ascii_str, count_digit, count_alpha, count_space, count_upper

__posts_glob_pattern = 'posts/*.json'
__file_path_template = 'posts_txt/p{}.txt'
__comment_path_template = 'comments/c{}.json'
__out_file_path = 'tmp/data.tsv'
__comment_count_file_path = 'tmp/comment_count.txt'
__pid_len_max = 6


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
    return [__pid(p) for p in paths]


def __comment_count(file_path):
    with open(file_path, 'rt') as f:
        jo = json.load(f)
        res = jo['total']
        return int(res)


def __save_dict(d, file_path):
    with open(file_path, 'wt') as out:
        pprint(d, stream=out)


def __token_count(s):
    return len(s.split())


def __token_length_mean(s):
    return np.mean([len(w) for w in s.split()])


def __pid(file_path):
    return path.basename(file_path)[1:-5]


def __pid_padded(pid):
    return str(pid).zfill(__pid_len_max)


def __main():
    paths = glob.glob(__posts_glob_pattern)
    pids = __pids(paths)
    df = pd.DataFrame(
        columns=['comment_count',
                 'digit_char_ratio',
                 'char_count',
                 'token_count',
                 'token_length_mean',
                 'author',
                 'text'
                 ],
        index=pids
    )
    n = 0
    pid_to_comment_count = dict()
    for p in paths:
        with open(p, 'rt') as f:
            pid = __pid(p)
            n += 1
            comment_path = __comment_path_template.format(pid)
            comment_count = __comment_count(comment_path)
            pid_to_comment_count[__pid_padded(pid)] = comment_count
            print(f'n={n}, pid={repr(pid)}, comment_count={repr(comment_count)}')
            jo = json.load(f)
            title = __title(jo)
            content = __content(jo)
            author = __author(jo)
            text = '{} {}'.format(title, content)
            char_count = len(text)
            digit_char_ratio = count_digit(text) / char_count
            token_count = __token_count(text)
            token_length_mean = __token_length_mean(text)
            df.loc[pid] = pd.Series({
                'comment_count': comment_count,
                'digit_char_ratio': digit_char_ratio,
                'char_count': char_count,
                'token_count': token_count,
                'token_length_mean': token_length_mean,
                'author': author,
                'text': text
            })
            # file_path = file_path_template.format(pid)
            # __save_text_file(file_path, text)
    df = pd.get_dummies(df, columns=['author'], prefix=['author'])
    print('Saving dataframe...')
    df.to_csv(__out_file_path, sep='\t')
    print(f'Saved {__out_file_path}\nSaving pid_to_comment_count...')
    __save_dict(pid_to_comment_count, __comment_count_file_path)
    print(f'Saved {__comment_count_file_path}')


if __name__ == '__main__':
    __main()
