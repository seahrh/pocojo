import json
import re

with open('comments/c300046.json', 'rt') as f:
    c = json.load(f)
    # print(f'c={repr(c)}')
    comments = c['comments']
    print(f'c.length={repr(len(comments))}')
    for i, comment in enumerate(comments):
        comment_txt = re.sub('<[^<]+?>', '', comment['content'])
        print(f'#{i}\n{comment_txt}')
        for j, child in enumerate(comment['children']):
            child_txt = re.sub('<[^<]+?>', '', child['content'])
            print(f'#{i}.{j}\n{child_txt}')
