import logging
from feature_extractor import SparseFeatureExtractor
from sentence_batch import SentenceBatch
from parser_state import ParserState
from arc_standard_transition_system import ArcStandardTransitionSystem, \
    ArcStandardTransitionState

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
        self.feature_extractor = SparseFeatureExtractor(self.feature_strings,
                                                        self.feature_maps)

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

        # Populate feature outputs
        for i in range(self.batch_size):
            if self.state(i) == None:
                continue

            self.logger.debug('Slot(%d): extract features' % i)
            fvec = self.feature_extractor.extract(self.state(i))
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

        return features_major_types, features_output, gold_actions, \
               self.num_epochs
