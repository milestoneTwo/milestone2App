
class Pipeline(object):
    __instance = None
    steps = {}
    def __new__(cls):
        if Pipeline.__instance is None:
            Pipeline.__instance = object.__new__(cls)
        return Pipeline.__instance

    def __call__(cls, step):
        Pipeline.__instance.steps[f'{step.__name__}'] = step