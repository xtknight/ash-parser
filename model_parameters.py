'''
Contains all parameters for a model
'''
FILE_TYPES = ['label-map', 'word-map', 'tag-map', 'parser-config', 'embeddings',
              'feature-def', 'feature-bag-bin', 'training-corpus-hash',
              'trained-config']

def fixPath(m):
    if m.endswith('/'):
        return m[:-1]
    else:
        return m

class ModelParameters(object):
    def __init__(self, modelFolder):
        assert modelFolder != None

        self.modelFolder = modelFolder
        self.trainingFile = None
        #self.tuningFile = None
        self.testingFile = None
        self.cfg = None
        self.lexicon = None

    '''
    Returns the filename for the requested file type
    Corresponds to files in SyntaxNet context
    e.g., word-map, label-map
    '''
    def getFilePath(self, fileType):
        assert fileType in FILE_TYPES
        assert self.modelFolder != None
        return '%s/%s' % (fixPath(self.modelFolder), fileType)

    def isValidModel(self):
        return self.modelFolder != None and self.trainingFile != None and \
            self.testingFile != None
