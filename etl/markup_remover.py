from html import unescape
from html.parser import HTMLParser

# Based on https://stackoverflow.com/a/925630/519951


class HtmlStripper(HTMLParser):
    def error(self, message):
        pass

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_html(html):
    html = unescape(html)
    s = HtmlStripper()
    s.feed(html)
    return s.get_data()
