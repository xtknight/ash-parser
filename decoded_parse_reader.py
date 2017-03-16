import logging
from feature_extractor import SparseFeatureExtractor
from sentence_batch import SentenceBatch
from parser_state import ParserState
from arc_standard_transition_system import ArcStandardTransitionSystem, \
    ArcStandardTransitionState

'''
Provide a batch of decoded sentences to the trainer

Maintains batch_size slots of sentences, each one with its own parser state
'''
class DecodedParseReader(object):
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
            self.parser_states[i] = ParserState(self.sentence_batch.sentence(i),
                                                self.feature_maps)
            # necessary for initializing and pushing root
            # keep arc_states in sync with parser_states
            self.arc_states[i] = \
                ArcStandardTransitionState(self.parser_states[i])
        else:
            self.parser_states[i] = None
            self.arc_states[i] = None
        if self.state(i) != None:
            self.docids_.insert(0, self.state(i).sentence().docid())

    '''
    Perform the next best decoded action for each state

    scores[i][k]: probability of each action k for token state i
                  as far as I know, raw logits

    filled_count: number of items of scores filled (if 0, forces SHIFT
                  for the first time). otherwise, should be greater than 0 an
                  less than or equal to batch_size
    '''
    def performActions(self, scores, filled_count):
        for batch_index in range(self.batch_size):
            state = self.state(batch_index)
            if state != None:
                # default action if none given
                bestAction = ParserState.SHIFT
                bestScore = float('-inf')

                # check to make sure decisions are filled for this batch i
                if filled_count > batch_index:
                    # look through top k estimated transition actions and
                    # pick most suitable one
                    for action in range(len(scores[batch_index])):
                        score = scores[batch_index][action]
                        if self.transition_system \
                                .isAllowedAction(action, state):
                            self.logger.debug('Slot(%d): action candidate:'
                                ' %s, score=%.8f - allowed' % (batch_index, \
                                self.transition_system.actionAsString(
                                    action, state,
                                    self.feature_maps), score))

                            if score > bestScore:
                                bestAction = action
                                bestScore = score
                        else:
                            self.logger.debug('Slot(%d): action candidate:'
                                ' %s, score=%.8f - unallowed' % (batch_index, \
                                self.transition_system.actionAsString(
                                    action, state,
                                    self.feature_maps), score))

                self.logger.debug('Slot(%d): perform action %s, score=%.8f' %
                    (batch_index, self.transition_system.actionAsString(
                        bestAction, state, \
                        self.feature_maps), bestScore))

                try:
                    self.transition_system.performAction(
                        bestAction, state)
                except:
                    self.logger.debug(
                        'Slot(%d): invalid action at batch slot' % batch_index)

                    self.transition_system.performAction(
                        action=self.transition_system.getDefaultAction(
                            state), state=state)

                if self.transition_system.isFinalState(state):
                    #self.computeTokenAccuracy(state)
                    self.sentence_map_ \
                        [state.sentence().docid()] = state.sentence()

                    self.logger.debug('Slot(%d): final state reached' \
                        % batch_index)

                    self.addParseToDocument(state, True, \
                        self.sentence_map_[state.sentence().docid()])

    '''
    Concatenate and return feature bags for all sentence slots, grouped
    by feature major type

    Returns (None, None, None, ...) if no sentences left
    '''
    def nextFeatureBags(self, scores, filled_count):
        self.performActions(scores, filled_count)

        for i in range(self.batch_size):
            if self.state(i) == None:
                continue

            while(self.transition_system.isFinalState(self.state(i))):
                self.logger.debug('Advancing sentence %d' % i)
                self.advanceSentence(i)
                if self.state(i) == None:
                    break

        if self.sentence_batch.size() == 0:
            self.num_epochs += 1
            if self.epoch_print:
                self.logger.info('Starting epoch %d' % self.num_epochs)
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

        return features_major_types, features_output, self.num_epochs, \
               filled_count

    '''
    Adds transition state specific annotations to the document
    '''
    def addParseToDocument(self, state, rewrite_root_labels, sentence):
        for i in range(state.numTokens()):
            token = sentence.mutableToken(i)
            token.setParsedLabel(
                self.feature_maps['label'].indexToValue(state.label(i)))

            if (state.head(i) != -1):
                token.setParsedHead(state.head(i))
            else:
                token.clearParsedHead()
                if rewrite_root_labels:
                    token.setParsedLabel('ROOT')

    '''
    Concatenate and return sentence annotations for all sentence slots

    Returns (None, None, None, ...) if no sentences left
    '''
    def getNextAnnotations(self):
        sentences = []
        while (len(self.docids_) > 0) and \
                (self.docids_[-1] in self.sentence_map_):

            self.logger.debug('Sentence(%d): %s' % (self.docids_[-1], \
                str(self.sentence_map_[self.docids_[-1]].tokens)))

            sentences.append(self.sentence_map_[self.docids_[-1]])
            del self.sentence_map_[self.docids_[-1]]
            self.docids_.pop()

        return sentences
