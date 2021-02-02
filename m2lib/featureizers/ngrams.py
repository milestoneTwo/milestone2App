from m2lib.pickler.picklable import PickleDef, Picklable

class Ngramer(Picklable):
    def __init__(self):
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super(Ngramer, self).__init__(**self.pickle_kwargs)

    def save(self):
        super().save()

    def load(self):
        self.__dict__ = super(Ngramer, self).load()