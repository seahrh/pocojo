import time
import math


class Timer:

    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def seconds_to_hhmmss(secs):
    secs = int(secs)
    hh = int(math.floor(secs / 3600))
    mm = int(math.floor((secs - (hh * 3600)) / 60))
    ss = secs - (hh * 3600) - (mm * 60)
    return '{:2d}:{:2d}:{:2d}'.format(hh, mm, ss)
