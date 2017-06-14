

class Dataset(object):
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w

    def next_batch(self):
        # generator for the data
        return ((x, y, w) for (x, y, w) in zip(self.x, self.y, self.w))