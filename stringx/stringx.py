from unicodedata import normalize


def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value  # Instance of str


def to_ascii_str(u):
    """Normalise (normalize) unicode data in Python to remove umlauts, accents etc.

    Based on https://gist.github.com/j4mie/557354
    """
    return to_str(normalize("NFKD", u).encode('ASCII', 'ignore'))
