import glob
import json
import numpy as np

comments_glob_pattern = 'comments/*.json'

ns = []
paths = glob.glob(comments_glob_pattern)
for p in paths:
    with open(p, 'rt') as f:
        jo = json.load(f)
        comments = jo['comments']
        n = 0
        for i, comment in enumerate(comments):
            n += 1
            for j, child in enumerate(comment['children']):
                n += 1
        ns.append(n)
res = np.percentile(ns, [25, 50, 75, 80, 90, 100])
print(f'res={repr(res)}')
