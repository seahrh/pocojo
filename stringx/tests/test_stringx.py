import unittest
from stringx.stringx import *


class TestStringx(unittest.TestCase):

    def test_is_number(self):
        self.assertEqual(is_number(''), False)
        self.assertEqual(is_number('1970'), True)
        self.assertEqual(is_number(1970), True)
        self.assertEqual(is_number('2am'), False)
        self.assertEqual(is_number('a'), False)
        self.assertEqual(is_number('NaN'), False)
        self.assertEqual(is_number('nan'), False)
        self.assertEqual(is_number('1'), True)
        self.assertEqual(is_number('-1'), True)
        self.assertEqual(is_number('1.1'), True)
        self.assertEqual(is_number('-1.1'), True)


if __name__ == '__main__':
    unittest.main()
