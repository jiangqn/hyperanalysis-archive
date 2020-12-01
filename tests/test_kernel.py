
class Kernel(object):

    def __init__(self):
        super(Kernel, self).__init__()
        self.k = 10

    def __call__(self, x):
        print(x)
        print(self.k)
        return x

k = Kernel()

print(k(1))