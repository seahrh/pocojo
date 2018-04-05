import unittest
from etl.markup_remover import strip_html


class TestMarkupRemover(unittest.TestCase):

    def test_strip_html(self):
        self.assertEqual(strip_html(''), '')
        self.assertEqual(strip_html('ab'), 'ab')
        self.assertEqual(strip_html('<p>p</p>'), 'p')
        self.assertEqual(strip_html('<p class="blue">p</p>'), 'p')
        self.assertEqual(strip_html('a&;b'), 'a&;b')
        self.assertEqual(strip_html('a &; b'), 'a &; b')
        self.assertEqual(strip_html('a &amp; b'), 'a & b')


if __name__ == '__main__':
    unittest.main()
