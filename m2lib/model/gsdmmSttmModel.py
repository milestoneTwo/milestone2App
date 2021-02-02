from gsdmm import MovieGroupProcess
from m2lib.pickler.picklable import Picklable, PickleDef
from m2lib.readers.readdata import Read
from m2lib.featureizers.preprocessor import Preprocessor

class GSDMM(Picklable):
    def __init__(self):
        self.model = None
        self.gsdmm_args = {
            'K': 10,
            'alpha': 0.1,
            'beta': 0.1,
            'n_iters': 30
        }
        pd = PickleDef(self)
        super(GSDMM, self).__init__(**pd())

    def train_model(self, corpus):
        mgp = MovieGroupProcess(**self.gsdmm_args)
        vocab = set(x for doc in corpus for x in doc)
        n_terms = len(vocab)
        model = mgp.fit(corpus, n_terms)
        self.model = model
        self.save()

    def save(self, obj=None):
        super(GSDMM, self).save()
        pass

    def load(self):
        super(GSDMM, self).load()

if __name__ == '__main__':
    read = Read(file='WikiLarge_Train.csv')
    train_set = read.read_dfs['WikiLarge_Train.csv']['original_text'][:20]
    preprocessor = Preprocessor()
    # preprocessor.train_phrase_model(train_set)
    processed = preprocessor.pipeline(train_set)
    gsdmm = GSDMM()
    gsdmm.train_model(preprocessor.corpus_)
    print(gsdmm.model)



