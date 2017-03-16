'''
Represents a lexicon, which describes all encountered tokens and frequencies,
along with unknown tokens.

The lexicon is typically computed during training time.
'''

from utils import *
from conll_utils import ParsedConllFile, ParsedConllSentence, ParsedConllToken
from feature_map import IndexEncodedFeatureMap

class Lexicon(object):
    def __init__(self, modelParams):
        self.modelParams = modelParams

        self.featureMaps = None

        self.tagMap = IndexEncodedFeatureMap()
        self.labelMap = IndexEncodedFeatureMap()
        self.wordMap = IndexEncodedFeatureMap()


    '''
    Compute a lexicon (using the training data)
    '''
    def compute(self):
        trainingData = ParsedConllFile()
        trainingData.read(open(self.modelParams.trainingFile, 'r',
                          encoding='utf-8').read())
        for sentence in trainingData:
            for token in sentence.tokens:
                # for SyntaxNet,
                # normalization ONLY happens in lexicon builder
                # yet numbers and up as <UNKNOWN> during training
                # interesting...
                form = normalizeDigits(token.FORM)

                self.wordMap.incrementTerm(form)
                self.tagMap.incrementTerm(token.XPOSTAG)
                self.labelMap.incrementTerm(token.DEPREL)
        
        self.finalizeLexicon()


    def read(self):
        self.tagMap = IndexEncodedFeatureMap().loadFrom(
            self.modelParams.getFilePath('tag-map'))
        self.labelMap = IndexEncodedFeatureMap().loadFrom(
            self.modelParams.getFilePath('label-map'))
        self.wordMap = IndexEncodedFeatureMap().loadFrom(
            self.modelParams.getFilePath('word-map'))
        
        # special values don't get saved, so we still need to finalize lexicon
        self.finalizeLexicon()


    def write(self):
        self.tagMap.writeTo(self.modelParams.getFilePath('tag-map'))
        self.labelMap.writeTo(self.modelParams.getFilePath('label-map'))
        self.wordMap.writeTo(self.modelParams.getFilePath('word-map'))

    '''
    After done reading corpus...
    '''
    def finalizeLexicon(self):
        self.wordMap.finalizeBaseValues()
        self.tagMap.finalizeBaseValues()
        self.labelMap.finalizeBaseValues()

        # order of special tokens matches SyntaxNet

        self.wordMap.appendSpecialValue("<UNKNOWN>")
        self.tagMap.appendSpecialValue("<UNKNOWN>")
        self.labelMap.appendSpecialValue("<UNKNOWN>")

        self.wordMap.appendSpecialValue("<OUTSIDE>")
        self.tagMap.appendSpecialValue("<OUTSIDE>")
        self.labelMap.appendSpecialValue("<OUTSIDE>")

        # FIXME: is <ROOT> in tag even possible? it seemed to happen in
        # testdata but not in UD_English
        # difference between stack.tag and stack.token.tag?
        #self.tagMap.appendSpecialValue("<ROOT>")
        self.labelMap.appendSpecialValue("<ROOT>")

        self.featureMaps = {'word': self.wordMap, 'tag': self.tagMap,
                            'label': self.labelMap}


    def getFeatureMaps(self):
        assert self.featureMaps != None, 'feature maps not yet created'
        return self.featureMaps
