import string
from unicodedata import normalize


def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value  # Instance of str


def strip_punctuation(s):
    """This uses the 3-argument version of str.maketrans with arguments (x, y, z) where 'x' and 'y'
    must be equal-length strings and characters in 'x'
    are replaced by characters in 'y'. 'z'
    is a string (string.punctuation here)
    where each character in the string is mapped
    to None
    translator = str.maketrans('', '', string.punctuation)

    This is an alternative that creates a dictionary mapping
    of every character from string.punctuation to None (this will
    also work)

    Based on https://stackoverflow.com/a/34294398/519951
    """
    translator = str.maketrans(dict.fromkeys(string.punctuation))
    return s.translate(translator)


def to_ascii_str(u):
    """Normalise (normalize) unicode data in Python to remove umlauts, accents etc.

    Based on https://gist.github.com/j4mie/557354
    """
    return to_str(normalize("NFKD", u).encode('ASCII', 'ignore'))


def is_number(s):
    try:
        float(s)
    except ValueError:
        return False
    return True
