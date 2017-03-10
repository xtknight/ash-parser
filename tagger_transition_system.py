# Translation of tagger_transitions.cc from SyntaxNet

from building_blocks import PosTaggedComponent, DepRelTaggedWord, DepRelTaggedPhrase
from parser_state import ParserState

class TaggerTransitionSystem(object):
    def __init__(self):
        pass

    def shiftAction(self, tag):
        return tag

    def numActionTypes(self):
        return 1

    def numActions(self, numLabels):
        return numLabels ## TODO

    def getDefaultAction(self, state):
        return self.shiftAction(0)

    def getNextGoldAction(self, state):
        if not state.endOfInput():
            return self.shiftAction(self.transitionState(state).goldTag(state.next()))

        return self.shiftAction(0)

    def isAllowedAction(self, action, state):
        return (not state.endOfInput())

    def performAction(self, action, state):
        # apparently history is not implemented in SyntaxNet either?
        self.performActionWithoutHistory(action, state)

    def performActionWithoutHistory(self, action, state):
        assert not state.endOfInput()
        if not state.endOfInput:
            self.mutableTransitionState(state).setTag(state.next(), action)
            state.push(state.next())
            state.advance()

    def isFinalState(self, state):
        return state.endOfInput()

    def actionAsString(self, action, state, feature_maps):
        return 'SHIFT(' + feature_maps['tag'].indexToValue(action) + ')'

    def isDeterministicState(self, state):
        return False
