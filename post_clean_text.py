import glob
import json
from os import path

from etl.markup_remover import strip_html
from stringx.stringx import to_ascii_str

posts_glob_pattern = 'posts/*.json'
file_path_template = 'posts_txt/p{}.txt'

n = 0
paths = glob.glob(posts_glob_pattern)
for p in paths:
    with open(p, 'rt') as f:
        jo = json.load(f)
        posts = jo['posts']
        if not posts:
            raise AssertionError('posts must not be empty')
        post = posts[0]
        title = post['title']
        if not title:
            raise AssertionError('title must not be empty')
        content = post['content']
        if not content:
            raise AssertionError('content must not be empty')
        s = '{} {}'.format(title, content)
        s = to_ascii_str(strip_html(s))
        # print(f'{repr(s)}\n')
        pid = path.basename(p)[1:-5]
        file_path = file_path_template.format(pid)
        with open(file_path, 'wt', encoding='utf8') as out:
            out.write(s)
        n += 1
        print(f'n={n}, pid={pid}')
