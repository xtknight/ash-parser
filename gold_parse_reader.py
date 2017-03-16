import logging
from feature_extractor_opt import SparseFeatureExtractor
from sentence_batch import SentenceBatch
from parser_state import ParserState
from arc_standard_transition_system import ArcStandardTransitionSystem, \
    ArcStandardTransitionState
from multiprocessing import Pool

feature_extractor = None

def f(s):
    #fd = open('/tmp/featext.bin', 'rb')
    #import copy
    #my_feature_extractor = copy.deepcopy(feature_extractor)
    #fd.close()
    return feature_extractor.extract(s, doLogging=False)

def batchExtractSparse(allStates, poolCount):
    with Pool(poolCount) as p:
        result = p.map(f, allStates)
    return result



'''
Provide a batch of sentences to the trainer

Maintains batch_size slots of sentences, each one with its own parser state
'''
class GoldParseReader(object):
    def __init__(self, input_corpus, batch_size, feature_strings, feature_maps,
                 epoch_print = True):
        self.input_corpus = input_corpus
        self.batch_size = batch_size
        self.feature_strings = feature_strings
        self.feature_maps = feature_maps
        self.epoch_print = epoch_print

        global feature_extractor
        #feature_extractor = SparseFeatureExtractor(self.feature_strings,
        #                                           self.feature_maps)
        #import pickle
        #pickle.dump(feature_extractor, open('/tmp/featext.bin', 'wb'))

        self.feature_extractor = SparseFeatureExtractor(self.feature_strings,
                                                        self.feature_maps)
        feature_extractor = self.feature_extractor

        self.sentence_batch = SentenceBatch(input_corpus, self.batch_size)
        self.parser_states = [None for i in range(self.batch_size)]
        self.arc_states = [None for i in range(self.batch_size)]
        self.transition_system = ArcStandardTransitionSystem()
        self.logger = logging.getLogger('GoldParseReader')
        self.num_epochs = 0

    def state(self, i):
        assert i >= 0 and i < self.batch_size
        return self.parser_states[i]

    '''
    Advance the sentence for slot i
    '''
    def advanceSentence(self, i):
        self.logger.debug('Slot(%d): advance sentence' % i)
        assert i >= 0 and i < self.batch_size
        if(self.sentence_batch.advanceSentence(i)):
            self.parser_states[i] = ParserState(self.sentence_batch.sentence(i),
                                                self.feature_maps)
            # necessary for initializing and pushing root
            # keep arc_states in sync with parser_states
            self.arc_states[i] = \
                ArcStandardTransitionState(self.parser_states[i])
        else:
            self.parser_states[i] = None
            self.arc_states[i] = None

    '''
    Perform the next gold action for each state
    '''
    def performActions(self):
        for i in range(self.batch_size):
            if self.state(i) != None:
                self.logger.debug('Slot(%d): perform actions' % i)

                nextGoldAction = \
                    self.transition_system.getNextGoldAction(self.state(i))

                self.logger.debug('Slot(%d): perform action %s' %
                    (i, self.transition_system.actionAsString(
                        nextGoldAction, self.state(i), self.feature_maps)))

                try:
                    self.transition_system.performAction(
                        action=nextGoldAction,
                        state=self.state(i))
                except:
                    self.logger.debug(
                        'Slot(%d): invalid action at batch slot' % i)
                    # This is probably because of a non-projective input
                    # We could projectivize or remove it...
                    self.transition_system.performAction(
                        action=self.transition_system.getDefaultAction(
                            self.state(i)),
                        state=self.state(i))

    '''
    Concatenate and return feature bags for all sentence slots, grouped
    by feature major type

    Returns (None, None, None, ...) if no sentences left
    '''
    def nextFeatureBags(self):
        import time
        self.logger.info('nextFeatureBags -- START %.4f' % time.time())
        self.performActions()
        for i in range(self.batch_size):
            if self.state(i) == None:
                continue

            while(self.transition_system.isFinalState(self.state(i))):
                self.logger.debug('Advancing sentence ' + str(i))
                self.advanceSentence(i)
                if self.state(i) == None:
                    break

        if self.sentence_batch.size() == 0:
            self.num_epochs += 1
            if self.epoch_print:
                self.logger.info('Starting epoch ' + str(self.num_epochs))
            self.sentence_batch.rewind()
            for i in range(self.batch_size):
                self.advanceSentence(i)

        # a little bit different from SyntaxNet:
        # we don't support feature groups
        # we automatically group together the similar types
        # features_output = [[] for i in range(self.feature_strings)]
        features_major_types = None
        features_output = None
        gold_actions = None

        statesToExtract = []
        # Populate feature outputs
        for i in range(self.batch_size):
            if self.state(i) == None:
                continue
            statesToExtract.append(self.state(i))

        #import pickle
        #pickle.dump(statesToExtract[-1], open('/tmp/state.bin', 'wb'))

        self.logger.info('nextFeatureBags -- EXSTART %.4f' % time.time())
        #allBatches = batchExtractSparse(statesToExtract, 1)

        # Populate feature outputs
        for i in range(self.batch_size):
            if self.state(i) == None:
                continue

            self.logger.debug('Slot(%d): extract features' % i)
            fvec = self.feature_extractor.extract(self.state(i), doLogging=False)
            assert len(fvec.types) == len(self.feature_strings)
            major_types, ids = fvec.concatenateSimilarTypes()

            if features_output == None:
                features_major_types = [t for t in major_types]
                features_output = [[] for t in major_types]
            else:
                assert len(features_major_types) == len(major_types)
                assert len(features_output) == len(major_types)

            for k in range(len(features_major_types)):
                features_output[k] += ids[k]

        self.logger.info('nextFeatureBags -- EXEND %.4f' % time.time())
        
        '''
        z = 0
        # Populate feature outputs
        for i in range(self.batch_size):
            if self.state(i) == None:
                continue

            self.logger.debug('Slot(%d): extract features' % i)
            fvec = allBatches[z]
            assert len(fvec.types) == len(self.feature_strings)
            major_types, ids = fvec.concatenateSimilarTypes()

            if features_output == None:
                features_major_types = [t for t in major_types]
                features_output = [[] for t in major_types]
            else:
                assert len(features_major_types) == len(major_types)
                assert len(features_output) == len(major_types)

            for k in range(len(features_major_types)):
                features_output[k] += ids[k]
            z += 1
        '''

        # Fill in gold actions
        for i in range(self.batch_size):
            if self.state(i) != None:
                if gold_actions == None:
                    gold_actions = []

                try:
                    gold_actions.append(
                        self.transition_system.getNextGoldAction(self.state(i)))
                except:
                    self.logger.info('Warning: invalid batch slot')
                    ## TODO: remove erroneous ones from training set??
                    gold_actions.append(
                        self.transition_system.getDefaultAction(self.state(i)))

        self.logger.info('nextFeatureBags -- END %.4f' % time.time())

        return features_major_types, features_output, gold_actions, \
               self.num_epochs
'''
## performance tests

import pickle
statesToExtract = []
a=pickle.load(open('/tmp/state.bin', 'rb'))
for v in range(5000):
    statesToExtract.append(a)

feature_extractor = pickle.load(open('/tmp/featext.bin', 'rb'))

import time
startTime = time.time()

for pc in [1,2,3,4,5,6,7]:
    print('nextFeatureBags -- EXSTART PC(%d)' % pc)
    batchExtractSparse(statesToExtract, pc)
    endTime = time.time()
    print('nextFeatureBags -- EXEND PC(%d) %.4fs (%.2f states/sec)' % (pc, endTime-startTime, float(len(statesToExtract)) / (endTime-startTime)))
    #feature_extractor.extract(statesToExtract[-1])
'''
