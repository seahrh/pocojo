import math


def seconds_to_hhmmss(secs):
    secs = int(secs)
    hh = int(math.floor(secs / 3600))
    mm = int(math.floor((secs - (hh * 3600)) / 60))
    ss = secs - (hh * 3600) - (mm * 60)
    return '{:2d}:{:2d}:{:2d}'.format(hh, mm, ss)
