from gensim.models.coherencemodel import CoherenceModel
from m2lib.pickler.picklable import PickleDef, Picklable

class MetricsStore(Picklable):
    def __init__(self):
        self.dfs = None
        self.coherence = None
        self.perplexity = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(**self.pickle_kwargs)

    def save(self):
        super().save()

    def load(self):
        super().load()

class MetricsStore(Picklable):
    def __init__(self):
        self.dfs = None
        self.coherence = None
        self.perplexity = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(**self.pickle_kwargs)

    def pipeline(self, model):

    def save(self):
        super().save()

    def load(self):
        super().load()

