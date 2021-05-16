class SrotolaTransform:
    def __call__(self, x):
        res = x.view(-1)
        return res

class PercColoriTransform:
    def __call__(self, x):
        res = x.view(3, -1).sum(1)
        return res