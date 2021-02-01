import json
from types import SimpleNamespace

class PickleDef:
    def __init__(self, obj):
        self.name = obj.__class__.__name__
        self.fname = f'{obj.__class__.__name__}.pkl'
        self.obj = obj


    def __call__(self):
        return self.__dict__




if __name__ == '__main__':
    listy = [1,2,3,4]
    pn = PickleDef(listy)
    print(pn())

