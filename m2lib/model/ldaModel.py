from m2lib.pickler.picklable import Picklable, PickleDef
from m2lib.featureizers.preprocessor import Preprocessor, PreprocessorStore
from m2lib.readers.readdata import Read
from m2lib.featureizers.bowfeature import BOWFeature, BOWFeatureStore
from m2lib.featureizers.tfidfeature import TFIDFeature, TFIDFeatureStore
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaMulticore

class LDAModelStore(Picklable):
    def __init__(self):
        self.model = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(**self.pickle_kwargs)

    def save(self):
        super().save()

    def load(self):
        super().load()

class LDAModel(Picklable):
    def __init__(self, force_build=False):
        self.model = None
        self.lda_args = {
            'chunksize': 2000,
            'alpha' : 'auto',
            'eta': 'auto',
            'iterations' : 50,
            'num_topics' : 10,
            'passes' : 20,
            'eval_every' : None,
            # 'workers' : 3,
        }
        pd = PickleDef(self)
        super(LDAModel, self).__init__(force_build, **pd())

    def __call__(self, *args, **kwargs):
        pass

    def train_model(self, corpus, dictionary):
        print('training LDA model')
        if super().check_if_force_build():
            temp = dictionary[0]
            id2word = dictionary.id2token

            model = LdaModel(
                corpus=corpus,
                id2word=id2word,
                **self.lda_args
            )
            self.model = model
        else:
            pass

    def set_lda_args(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.lda_args:
                self.lda_args[k] = v
            else:
                raise KeyError

    def save(self, obj=None):
        super(LDAModel, self).save()
        pass

    def load(self):
        super(LDAModel, self).load()


class LDATFIDModelStore(Picklable):
    def __init__(self, force_build=False):
        self.model = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(force_build, **self.pickle_kwargs)

    def save(self):
        super().save()

    def load(self):
        super().load()


class LDATFIDModel(LDAModel, Picklable):
    def __init__(self, force_build=False):
        LDAModel.__init__(self)
        pd = PickleDef(self)
        Picklable.__init__(self, force_build, **pd())

    def train_model(self, corpus, dictionary):
        if super().check_if_force_build():
            super().train_model(corpus, dictionary)
        else:
            pass

    def set_lda_args(self, **kwargs):
        super().set_lda_args(**kwargs)

    def save(self, obj=None):
        super().save()
        pass

    def load(self):
        super().load()


if __name__ == '__main__':
    ps = PreprocessorStore()
    bowStore = BOWFeatureStore()

    tfid = TFIDFeature().pipeline(bowStore.corpus_, bowStore.dictionary)
    tfidStore = TFIDFeatureStore()
    tfidStore.corpus_ = tfid.corpus_
    tfidStore.dictionary = tfid.dictionary

    lda_model = LDAModel().train_model(bowStore.corpus_, bowStore.dictionary)
    lda_model_store = LDAModelStore()
    lda_model_store.model = lda_model.model

    lda_tfid_model = LDATFIDModel().train_model(tfidStore.corpus_, tfidStore.dictionary)
    lda_tfid_model_store = LDATFIDModelStore()
    lda_tfid_model_store.model = lda_tfid_model.model

    tfid = TFIDFeature()
    tfid.pipeline(bowStore.corpus_, bowStore.dictionary)
    tfidStore = TFIDFeatureStore()
    tfidStore.corpus_ = tfid.corpus_
    tfidStore.dictionary = tfid.dictionary
    tfidStore.save()

    lda_model = LDAModel()
    lda_model.train_model(bowStore.corpus_, bowStore.dictionary)
    lda_model_store = LDAModelStore()
    lda_model_store.model = lda_model.model
    lda_model_store.save()

    lda_tfid_model = LDATFIDModel()
    lda_tfid_model.train_model(tfidStore.corpus_, tfidStore.dictionary)
    lda_tfid_model_store = LDATFIDModelStore()
    lda_tfid_model_store.model = lda_tfid_model_store.model
    lda_tfid_model_store.save()






