import logging
from feature_extractor import SparseFeatureExtractor
from sentence_batch import SentenceBatch
from parser_state import ParserState
from arc_standard_transition_system import ArcStandardTransitionSystem, ArcStandardTransitionState

'''
Provide a batch of sentences to the trainer

Maintains batch_size slots of sentences, each one with its own parser state
'''
class DecodedParseReader(object):
    def __init__(self, input_corpus, batch_size, feature_strings, feature_maps, epoch_print = True):
        self.input_corpus = input_corpus
        self.batch_size = batch_size
        self.feature_strings = feature_strings
        self.feature_maps = feature_maps
        self.epoch_print = epoch_print
        self.feature_extractor = SparseFeatureExtractor(self.feature_strings, self.feature_maps)

        self.sentence_batch = SentenceBatch(input_corpus, self.batch_size)
        self.parser_states = [None for i in range(self.batch_size)]
        self.arc_states = [None for i in range(self.batch_size)]
        self.transition_system = ArcStandardTransitionSystem()
        self.logger = logging.getLogger('DecodedParseReader')
        self.num_epochs = 0

        self.docids_ = []
        # map docid to sentence
        self.sentence_map_ = dict()

    def state(self, i):
        assert i >= 0 and i < self.batch_size
        return self.parser_states[i]

    '''
    Advance the sentence for slot i
    '''
    def advanceSentence(self, i):
        assert i >= 0 and i < self.batch_size
        if(self.sentence_batch.advanceSentence(i)):
            self.parser_states[i] = ParserState(self.sentence_batch.sentence(i), self.feature_maps)
            # necessary for initializing and pushing root
            # keep arc_states in sync with parser_states
            self.arc_states[i] = ArcStandardTransitionState(self.parser_states[i])
        else:
            self.parser_states[i] = None
            self.arc_states[i] = None
        if self.state(i) != None:
            self.docids_.insert(0, self.state(i).sentence().docid())

    '''
    Perform the next gold action for each state
    Use this function for debugging
    This should decode perfectly

    (there may be a slight bug the second epoch because after the first we
    set gold root labels to 'ROOT', which is out of the vocabulary.
    it's confusing, because ROOT is different than <ROOT>)
    '''
    def performActions_GOLD(self, pred_top_k):
        for i in range(self.batch_size):
            if self.state(i) != None:
                nextGoldAction = self.transition_system.getNextGoldAction(self.state(i))
                self.logger.debug('performActions(): ' + str(i) + ': action ' + str(self.transition_system.actionAsString(nextGoldAction, self.state(i), self.feature_maps)))
                try:
                    self.transition_system.performAction(nextGoldAction, self.state(i))
                except:
                    self.logger.debug('Warning: performActions(): ' + str(i) + ': invalid action at batch slot')
                    # This is probably because of a non-projective input
                    # We could projectivize or remove it...
                    # Or just do default actions and give wrong data, like we do now...
                    ## TODO: remove erroneous ones from training set??
                    self.transition_system.performAction(self.transition_system.getDefaultAction(self.state(i)), self.state(i))

                if self.transition_system.isFinalState(self.state(i)):
                    #self.computeTokenAccuracy(self.state(i))
                    self.sentence_map_[self.state(i).sentence().docid()] = self.state(i).sentence()
                    self.logger.debug('addParseToDocument: ' + str(i))
                    self.addParseToDocument(self.state(i), True, self.sentence_map_[self.state(i).sentence().docid()])

    '''
    Perform the next best decoded action for each state

    pred_top_k[i]: top k actions (transition action integers) for token state i
    filled_count: number of items of array filled (will be 0, forcing SHIFT first time)
                  otherwise, should be greater than 0 and less than or equal to batch_size
    '''
    def performActions(self, pred_top_k, filled_count):
        #print('performActions: filled_count=%d, len(pred_top_k): %d' % (filled_count, len(pred_top_k)))

        for i in range(self.batch_size):
            if self.state(i) != None:
                nextGoldAction = self.transition_system.getNextGoldAction(self.state(i))
                self.logger.debug('Slot(%d) - Gold action: %s' % (i, self.transition_system.actionAsString(nextGoldAction, self.state(i), self.feature_maps)))

                #print('... state(%d): non-null' % i)
                #assert len(pred_top_k[i]) > 0
                bestAction = ParserState.SHIFT # default action if none given

                #if len(pred_top_k) > i:
                #    print('... len(pred_top_k[%d]): %d' % (i, len(pred_top_k[i])))

                # FIXME: why would state by non-null and yet the slot not be filled?

                if filled_count > i: # there may be none if we are in first state, which is an automatic SHIFT
                    # look through top k estimated transition actions and pick most suitable one
                    for k in range(len(pred_top_k[i])):
                        action = pred_top_k[i][k]
                        if self.transition_system.isAllowedAction(action, self.state(i)):
                            self.logger.debug('Slot(%d) - Possible action(%d): %s - allowed' % (i, k, self.transition_system.actionAsString(action, self.state(i), self.feature_maps)))
                        else:
                            self.logger.debug('Slot(%d) - Possible action(%d): %s - unallowed' % (i, k, self.transition_system.actionAsString(action, self.state(i), self.feature_maps)))

                    for k in range(len(pred_top_k[i])):
                        action = pred_top_k[i][k]
                        if self.transition_system.isAllowedAction(action, self.state(i)):
                            #if k != 0:
                            #    print('Picked alternate action ' + str(k) + ' successfully!')
                            #else:
                            #    print('Picked top 1 action ' + str(k) + ' successfully!')
                            bestAction = action
                            break

                self.logger.debug('performActions(): ' + str(i) + ': action ' + str(self.transition_system.actionAsString(bestAction, self.state(i), self.feature_maps)))
                try:
                    self.transition_system.performAction(bestAction, self.state(i))
                except:
                    self.logger.debug('Warning: performActions(): ' + str(i) + ': invalid action at batch slot')
                    # This is probably because of a non-projective input
                    # We could projectivize or remove it...
                    # Or just do default actions and give wrong data, like we do now...
                    ## TODO: remove erroneous ones from training set??
                    self.transition_system.performAction(self.transition_system.getDefaultAction(self.state(i)), self.state(i))

                if self.transition_system.isFinalState(self.state(i)):
                    #self.computeTokenAccuracy(self.state(i))
                    self.sentence_map_[self.state(i).sentence().docid()] = self.state(i).sentence()
                    self.logger.debug('addParseToDocument: ' + str(i))
                    self.addParseToDocument(self.state(i), True, self.sentence_map_[self.state(i).sentence().docid()])

    '''
    Concatenate and return feature bags for all sentence slots, grouped
    by feature major type

    Returns (None, None, None, ...) if no sentences left
    '''
    def nextFeatureBags(self, pred_top_k, filled_count):
        #if pred_top_k != None: ## FIXME: HACK
        self.performActions(pred_top_k, filled_count)

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

        filled_count = 0

        # Populate feature outputs
        for i in range(self.batch_size):
            if self.state(i) == None:
                continue

            # FIXME: the features extracted will be gold features, so beware
            # what we really need to do is set token parsed label, etc beforehand

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

            filled_count += 1

        return features_major_types, features_output, self.num_epochs, filled_count

    '''
    Adds transition state specific annotations to the document
    '''
    def addParseToDocument(self, state, rewrite_root_labels, sentence):
        for i in range(state.numTokens()):
            token = sentence.mutableToken(i)
            token.setParsedLabel(self.feature_maps['label'].indexToValue(state.label(i)))
            if (state.head(i) != -1):
                token.setParsedHead(state.head(i))
            else:
                token.clearParsedHead()
                if rewrite_root_labels:
                    token.setParsedLabel("ROOT")

    '''
    Concatenate and return sentence annotations for all sentence slots

    Returns (None, None, None, ...) if no sentences left
    '''
    def getNextAnnotations(self):
        sentences = []
        while (len(self.docids_) > 0) and (self.docids_[-1] in self.sentence_map_):
            self.logger.debug('Sentence[%d]: %s' % (self.docids_[-1], str(self.sentence_map_[self.docids_[-1]].tokens)))
            sentences.append(self.sentence_map_[self.docids_[-1]])
            del self.sentence_map_[self.docids_[-1]]
            self.docids_.pop()

        return sentences
