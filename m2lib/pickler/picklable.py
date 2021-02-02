from m2lib.pickler.pickler import Pickler

class PickleDef:
    def __init__(self, obj):
        self.name = obj.__class__.__name__
        self.fname = f'{obj.__class__.__name__}.pkl'
        self.obj = obj

    def __call__(self):
        return self.__dict__

class Picklable:
    def __init__(self, **pickle_kwargs):
        self.pickler = Pickler()
        self.pickle_kwargs = pickle_kwargs
        if self.__check_for_pickle():
            try:
                self.load()
            except Exception as e:
                print(e)
                pass
        else:
            print(f'no pickle exists for class {self.__class__.__name__}')
            print(f'object should implement super().save() to save')

    def __check_for_pickle(self):
        print('checking for pickle')
        return self.pickler.check_pickle(**self.pickle_kwargs)

    def save(self, obj=None):
        if obj:
            self.pickle_kwargs['obj'] = obj
        else:
            self.pickle_kwargs['obj'] = self.__dict__
        self.pickler.add_pickle(**self.pickle_kwargs)

    def load(self):
        print(f'{self.pickle_kwargs["name"]} loading from existing pickle')
        obj = self.pickler.get_pickle(**self.pickle_kwargs)
        if obj == False:
            print(f'{self.pickle_kwargs["name"]} obj not loaded from file')
        else:
            print(f'loading {self.pickle_kwargs["name"]} from file')
            # return obj
            self.__dict__ = obj


# TODO: Use this to make pickles classes and test
# class Child(Picklable):
#     def __init__(self):
#         pd = PickleDef(self)
#         self.pickle_kwargs = pd()
#         super(Child, self).__init__(**self.pickle_kwargs)
#
#     def save(self):
#         super().save()
#
#     def load(self):
#         self.__dict__.update(super(Child, self).load())


if __name__ == '__main__':
    # child = Child()
    # child.save()
    # print(child.dookie)
    # child.dookie = 'farts'
    # print(child.dookie)
    # child.save()
    # child = Child()
    # print(child.dookie)
else:
    pass
