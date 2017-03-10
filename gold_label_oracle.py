import logging
from parser_state import ParserState
from arc_standard_transition_system import ArcStandardTransitionSystem, ArcStandardTransitionState

'''
Reconstruct oracle transition sequence for each sentence
'''
class GoldLabelOracle(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.logger = logging.getLogger('GoldLabelOracle')

    '''
    Generates opcodes (tuples) to represent the entire transition sequence

    In this new version, the PREVIOUS state and NEXT GOLD action is returned

    STATE is ParserState object
    GOLD action is in form:
    (encoding used by ArcStandardTransitionSystem)

    Returns (STATE, GOLD)
    [GOLD action to take when in given STATE]
    '''
    def deriveNextGoldActionForState(self):
        for w in self.sentence.words:
            # make sure words are set properly WITH dependency information
            assert w.id != None
            assert w.fullword != None
            assert w.head != None
            assert w.deprel != None
            # components within words are optional in this case

        self.state = ParserState(self.sentence)
        arcState = ArcStandardTransitionState(self.state)
        transSystem = ArcStandardTransitionSystem()

        while not transSystem.isFinalState(self.state):
            # save previous state as-is
            previousState = copy.deepcopy(self.state)
            action = transSystem.getNextGoldAction(self.state)
            self.logger.debug('Performing action: ' + transSystem.actionAsString(action, self.state))
            assert transSystem.isAllowedAction(action, self.state)
            yield (previousState, action)
            transSystem.performActionWithoutHistory(action, self.state)
            self.logger.debug('Parser state: ' + arcState.toString(self.state))

    '''
    Get the buffer of all DepRelTaggedWord token instances in order

    Transitions should be replayed against this buffer and only this buffer
    '''
    def getBuffer(self):
        self.state = ParserState(self.sentence)
        return [self.state.getToken(i) for i in range(self.state.numTokens())]

    '''
    Inputs a sentence without dependency info and outputs one with dependency info
    after replaying specified transitions
    '''
    def replayTransitions(self, transitions):
        self.state = ParserState(self.sentence)

        for w in self.sentence.words:
            # make sure words are set properly WITHOUT dependency information
            assert w.id != None
            assert w.fullword != None
            assert w.head == None
            assert w.deprel == None
            # components within words are optional in this case

        arcState = ArcStandardTransitionState(self.state)
        transSystem = ArcStandardTransitionSystem()

        while (not transSystem.isFinalState(self.state)) and len(transitions) > 0:
            action = transitions[0] # get next action
            transitions = transitions[1:] # cut off for next transition

            self.logger.debug('Performing action: ' + transSystem.actionAsString(action, self.state))
            assert transSystem.isAllowedAction(action, self.state)

            transSystem.performActionWithoutHistory(action, self.state)
            self.logger.debug('Parser state: ' + arcState.toString(self.state))

        # we don't bother to look at token(-1) (the NULL 'ROOT' token)
        # this is not to be confused with the 'ROOT' deprel tag, which we will still use
        for i in range(0, self.state.numTokens()):
            # getToken(i) is just self.sentence.words[i]
            self.state.getToken(i).deprel = self.state.labelAsString(self.state.label(i))
            self.state.getToken(i).head = self.state.head(i)
            # NOTE: components within words don't need to be set (our role is to set dep tags, not pos tags)

    ## TODO
    '''
    TODO: FACTOR

    Inputs a sentence without dependency info and outputs one with dependency info
    after replaying specified transitions

    DOES NOT USE self.state
    '''
    def replayTransitionIndividual(self, transitions, state, arcState, transSystem):
        for w in self.sentence.words:
            # make sure words are set properly WITHOUT dependency information
            assert w.id != None
            assert w.fullword != None
            ## keep in mind these might already be tagged
            #assert w.head == None
            #assert w.deprel == None
            # components within words are optional in this case

        while (not transSystem.isFinalState(state)) and len(transitions) > 0:
            action = transitions[0] # get next action
            transitions = transitions[1:] # cut off for next transition

            self.logger.debug('Performing action: ' + transSystem.actionAsString(action, state))
            assert transSystem.isAllowedAction(action, state)

            transSystem.performActionWithoutHistory(action, state)
            self.logger.debug('Parser state: ' + arcState.toString(state))

        # we don't bother to look at token(-1) (the NULL 'ROOT' token)
        # this is not to be confused with the 'ROOT' deprel tag, which we will still use
        for i in range(0, state.numTokens()):
            # getToken(i) is just self.sentence.words[i]
            state.getToken(i).deprel = state.labelAsString(state.label(i))
            state.getToken(i).head = state.head(i)
            # NOTE: components within words don't need to be set (our role is to set dep tags, not pos tags)
