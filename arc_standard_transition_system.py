# Translation of arc_standard_transitions.cc from SyntaxNet

from conll_utils import ParsedConllSentence, ParsedConllToken
from parser_state import ParserState

class ArcStandardTransitionState(object):
    def __init__(self, state):
        state.push(-1) # ROOT node

    def toString(self, state):
        s = '['
        i = state.stackSize() - 1
        while i >= 0:
            word = state.getToken(state.stack(i)).FORM
            if i != state.stackSize() - 1:
                s += ' '

            # only for internal ROOT token at start of stack
            if word == None:
                s += 'ROOT'
            else:
                s += word

            i -= 1
        s += ']'

        i = state.next()
        while i < state.numTokens():
            s += ' ' + state.getToken(i).FORM
            i += 1
        return s

class ArcStandardTransitionSystem(object):
    def __init__(self):
        pass

    def shiftAction(self):
        return ParserState.SHIFT

    def leftArcAction(self, label):
        return 1 + (label << 1)

    def rightArcAction(self, label):
        return 1 + ((label << 1) | 1)

    def label(self, action):
        if action < 1:
            return -1
        else:
            return (action - 1) >> 1

    def actionType(self, action):
        if action < 1:
            return action
        else:
            return 1 + (~action & 1)

    def numActionTypes(self):
        return 3

    def numActions(self, numLabels):
        return 1 + 2 * numLabels

    def getDefaultAction(self, state):
        if(not state.endOfInput()):
            return self.shiftAction()

        return self.rightArcAction(2)

    def getNextGoldAction(self, state):
        # nothing else we can do except shift to the end
        # (leaving us with only the remaining 'ROOT' element)
        if(state.stackSize() < 2):
            assert not state.endOfInput()
            #print('STACK TOO SMALL: RETURN DEFAULT SHIFT ACTION')
            return self.shiftAction()

        # S|i|j
        # if HEAD(j) == i... (and we are done with children to the right of j)
        # add right arc i->j:
        # (S|i|j, B) => (S|i, B)
        if(state.goldHead(state.stack(0)) == state.stack(1) and \
                self.doneChildrenRightOf(state, state.stack(0))):
            gold_label = state.goldLabel(state.stack(0))
            return self.rightArcAction(gold_label)

        # S|i|j
        # if HEAD(i) == j... add left arc j->i:
        # (S|i|j, B) => (S|j, B)
        if(state.goldHead(state.stack(1)) == state.top()):
            gold_label = state.goldLabel(state.stack(1))
            return self.leftArcAction(gold_label)

        # S|i
        # nothing else we can do except shift to the next required arc
        # (S, i|B) => (S|i, B)
        return self.shiftAction()

    def doneChildrenRightOf(self, state, head):
        index = state.next()
        #num_tokens = state.sentence().token_size() ## ??
        num_tokens = state.numTokens()

        while(index < num_tokens):
            actual_head = state.goldHead(index)
            if(actual_head == head):
                return False

            if(actual_head > index):
                index = actual_head
            else:
                index += 1

        return True

    def isAllowedAction(self, action, state):
        if(self.actionType(action) == ParserState.SHIFT):
            return self.isAllowedShift(state)
        elif(self.actionType(action) == ParserState.LEFT_ARC):
            return self.isAllowedLeftArc(state)
        elif(self.actionType(action) == ParserState.RIGHT_ARC):
            return self.isAllowedRightArc(state)
        else:
            assert None

    def isAllowedShift(self, state):
        return (not state.endOfInput())

    def isAllowedLeftArc(self, state):
        # Left-arc requires two or more tokens on the stack but the first token
        # is the root and we do not want a left arc to the root.
        return (state.stackSize() > 2)

    def isAllowedRightArc(self, state):
        # Right arc requires three or more tokens on the stack.
        return (state.stackSize() > 1)

    def performAction(self, action, state):
        self.performActionWithoutHistory(action, state)

    def performActionWithoutHistory(self, action, state):
        if self.actionType(action) == ParserState.SHIFT:
            self.performShift(state)
        elif self.actionType(action) == ParserState.LEFT_ARC:
            self.performLeftArc(state, self.label(action))
        elif self.actionType(action) == ParserState.RIGHT_ARC:
            self.performRightArc(state, self.label(action))
        else:
            assert(None)

    def performShift(self, state):
        assert self.isAllowedShift(state)
        state.push(state.next())
        state.advance()

    def performLeftArc(self, state, label):
        assert self.isAllowedLeftArc(state)
        s0 = state.pop()
        state.addArc(state.pop(), s0, label)
        state.push(s0)

    def performRightArc(self, state, label):
        assert self.isAllowedRightArc(state)
        s0 = state.pop()
        s1 = state.pop()
        state.addArc(s0, s1, label)
        state.push(s1)

    def isDeterministicState(self, state):
        return state.stackSize() < 2 and (not state.endOfInput())

    def isFinalState(self, state):
        return state.endOfInput() and (state.stackSize() < 2)

    def actionAsTuple(self, action):
        if(self.actionType(action) == ParserState.SHIFT):
            return (ParserState.SHIFT,)
        elif(self.actionType(action) == ParserState.LEFT_ARC):
            return (ParserState.LEFT_ARC, self.label(action))
        elif(self.actionType(action) == ParserState.RIGHT_ARC):
            return (ParserState.RIGHT_ARC, self.label(action))
        else:
            return None

    def actionAsString(self, action, state, feature_maps):
        if(self.actionType(action) == ParserState.SHIFT):
            return 'SHIFT'
        elif(self.actionType(action) == ParserState.LEFT_ARC):
            return 'LEFT_ARC(' + \
                feature_maps['label'].indexToValue(self.label(action)) + ')'
        elif(self.actionType(action) == ParserState.RIGHT_ARC):
            return 'RIGHT_ARC(' + \
                feature_maps['label'].indexToValue(self.label(action)) + ')'
        else:
            return 'UNKNOWN'
