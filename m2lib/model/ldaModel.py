from m2lib.pickler.picklable import Picklable, PickleDef
from gensim.models import LdaModel

class LDAModel(Picklable):
     def __init__(self):
         pd = PickleDef(self)
         super(LDAModel, self).__init__(**pd())

     def __call__(self, *args, **kwargs):
         pass

     def train_model(self):
         num_topics = 10
         chunksize = 2000
         passes = 20
         iterations = 400
         eval_every = None



     def save(self, obj=None):
         super(LDAModel, self).save()
         pass

     def load(self):
         super(LDAModel, self).load()



if __name__ == '__main__':
    lda_model = LDAModel()
    lda_model.test = 'this is another test'
    lda_model.save()
    lda_model = LDAModel()
    print(lda_model.test)
