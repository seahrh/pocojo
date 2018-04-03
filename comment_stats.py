import glob
import json
import numpy as np

comments_glob_pattern = 'comments/*.json'


def comment_count(jo):
    comments = jo['comments']
    n = 0
    for i, comment in enumerate(comments):
        n += 1
        for _ in comment['children']:
            n += 1
    return n


ns = []
paths = glob.glob(comments_glob_pattern)
for p in paths:
    with open(p, 'rt') as f:
        _jo = json.load(f)
        ns.append(comment_count(_jo))
res = np.percentile(ns, [25, 50, 75, 80, 90, 100])
print(f'res={repr(res)}')
