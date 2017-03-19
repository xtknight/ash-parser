# translated from https://github.com/yahoo/YaraParser/blob/master/src/YaraParser/TransitionBasedSystem/Parser/Actions.java
# https://github.com/yahoo/YaraParser/blob/master/src/YaraParser/TransitionBasedSystem/Configuration/GoldConfiguration.java
# https://github.com/yahoo/YaraParser/blob/master/src/YaraParser/TransitionBasedSystem/Trainer/ArcEagerBeamTrainer.java

from conll_utils import ParsedConllSentence, ParsedConllToken
from parser_state import ParserState

class ArcEagerTransitionState(object):
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

class ArcEagerTransitionSystem(object):
    SHIFT       = 0
    REDUCE      = 1
    LEFT_ARC    = 2
    RIGHT_ARC   = 3

    Transitions = [SHIFT, REDUCE, LEFT_ARC, RIGHT_ARC]
    
    def __init__(self):
        assert self.actionType(0) == 0
        assert self.actionType(1) == 1
        assert self.actionType(2) == 2
        assert self.actionType(3) == 3
        assert self.actionType(4) == 2
        assert self.actionType(5) == 3
        assert self.actionType(6) == 2
        assert self.actionType(7) == 3
        self.didLastReduce = False
        pass

    def shiftAction(self):
        return ArcEagerTransitionSystem.SHIFT

    def reduceAction(self):
        return ArcEagerTransitionSystem.REDUCE

    def leftArcAction(self, label):
        return 2 + (label << 1)

    def rightArcAction(self, label):
        return 2 + ((label << 1) | 1)

    def label(self, action):
        if action < 2:
            return -1
        else:
            return (action - 2) >> 1

    def actionType(self, action):
        if action < 2:
            return action
        else:
            return 2 + (~(action-1) & 1)

    def numActionTypes(self):
        return 4

    def numActions(self, numLabels):
        return 2 + 2 * numLabels

    def getDefaultAction(self, state):
        if(not state.endOfInput()):
            return self.shiftAction()

        return self.rightArcAction(2)

    # http://cl.indiana.edu/~md7/nasslli10/02/02-transition.pdf
    '''
    def getNextGoldAction(self, state):
        # nothing else we can do except shift to the end
        # (leaving us with only the remaining 'ROOT' element)
        if(state.stackSize() < 2):
            assert not state.endOfInput()
            return self.shiftAction()

        # S|i|j
        # if HEAD(j) == i... (and we are done with children to the right of j)
        # add right arc i->j:
        # (S|i, j|B) => (S|i|j, B)
        #if(state.goldHead(state.stack(0)) == state.stack(1) and \
        #        self.doneChildrenRightOf(state, state.stack(0))):
        #    gold_label = state.goldLabel(state.stack(0))
        #    return self.rightArcAction(gold_label)

        if(state.goldHead(state.input(0)) == state.stack(0)):
            gold_label = state.goldLabel(state.input(0))
            return self.rightArcAction(gold_label)

        # S|i|j
        # if HEAD(i) == j... add left arc j->i:
        # (S|i, j|B) => (S, j|B)
        if(state.goldHead(state.stack(0)) == state.input(0)):
            gold_label = state.goldLabel(state.stack(0))
            return self.leftArcAction(gold_label)

        # S|i
        # (S|i, B) => (S, B)
        if(self.doneChildrenRightOf(state, state.stack(0))):
            return self.reduceAction()

        # S|i
        # nothing else we can do except shift to the next required arc
        # (S, i|B) => (S|i, B)
        return self.shiftAction()
    '''

    '''
    Arc Standard
    def getNextGoldAction(self, state):
        # nothing else we can do except shift to the end
        # (leaving us with only the remaining 'ROOT' element)

        # o0 = state.stack(0)
        # o1 = state.stack(1)

        if(state.stackSize() < 2):
            assert not state.endOfInput()
            #print('STACK TOO SMALL: RETURN DEFAULT SHIFT ACTION')
            return self.shiftAction()

        # S|i|j
        # if HEAD(i) == j... add left arc j->i:
        # (S|i|j, B) => (S|j, B)
        if(state.goldHead(o1) == o0 and \
                self.doneChildrenRightOf(state, o1)):
            gold_label = state.goldLabel(o1)
            return self.leftArcAction(gold_label)

        # S|i|j
        # if HEAD(j) == i... (and we are done with children to the right of j)
        # add right arc i->j:
        # (S|i|j, B) => (S|i, B)
        if(state.goldHead(o0) == o1 and \
                self.doneChildrenRightOf(state, o0)):
            gold_label = state.goldLabel(o0)
            return self.rightArcAction(gold_label)

        # S|i
        # nothing else we can do except shift to the next required arc
        # (S, i|B) => (S|i, B)
        return self.shiftAction()
    '''

    # https://github.com/yahoo/YaraParser/blob/master/src/YaraParser/TransitionBasedSystem/Trainer/ArcEagerBeamTrainer.java
    # staticOracle
    # http://demo.clab.cs.cmu.edu/fa2015-11711/images/b/b1/TbparsingSmallCorrection.pdf
    def getNextGoldAction(self, state):
        # nothing else we can do except shift to the end
        # (leaving us with only the remaining 'ROOT' element)

        ## FIXME: need this?
        #if(state.stackSize() < 2):
        #    assert not state.endOfInput()
        #    print('return S action', self.shiftAction())
        #    return self.shiftAction()

        #if state.endOfInput():
        #    self.didLastReduce = True
        #    if(state.stackSize() > 0):
        #        return self.reduceAction() ## TODO: necessary?

        B0 = state.input(0)
        o0 = state.stack(0)

        if o0 == -1:
            # reset this variable for this transition system
            self.didLastReduce = False

        print('o0<%s>:' % state.getToken(o0).FORM, o0)
        print('B0<%s>:' % state.getToken(B0).FORM, B0)

        #if(state.stackSize() < 2):
        #    assert not state.endOfInput()
        #    #print('STACK TOO SMALL: RETURN DEFAULT SHIFT ACTION')
        #    return self.shiftAction()

        print('state.goldHead(o0):', state.goldHead(o0))
        print('state.goldHead(B0):', state.goldHead(B0))

        print(o0, state.rootLabel(), state.goldHead(o0), B0)

        if(o0 != state.rootLabel() and state.goldHead(o0) == B0):
            gold_label = state.goldLabel(o0)
            print('return L-A action', self.leftArcAction(gold_label))
            return self.leftArcAction(gold_label)

        #??? needed?
        if(B0 != state.rootLabel() and state.goldHead(B0) == o0):
            gold_label = state.goldLabel(B0)
            print('return R-A action', self.rightArcAction(gold_label))
            return self.rightArcAction(gold_label)

        print("TODO: add root node")
        #print("cond1", o0 != state.rootLabel())
        #print("cond2", self.doneChildrenRightOf(state, o0))
        #print("cond3", state.hasHead(o0))

        ## NOTE: we use dummy root node unlike this pdf
        if(o0 != state.rootLabel() and self.doneChildrenRightOf(state, o0)
                and state.hasHead(o0)):
            print('return R action', self.reduceAction())
            return self.reduceAction()
        #elif(o0 == state.rootLabel()):
        #    print('return R action', self.reduceAction())
        #    return self.reduceAction()

        # S|i
        # nothing else we can do except shift to the next required arc
        # (S, i|B) => (S|i, B)
        print('return S action', self.shiftAction())
        return self.shiftAction()

    def doneChildrenRightOf(self, state, head):
        index = state.next()
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
        if(self.actionType(action) == ArcEagerTransitionSystem.SHIFT):
            return self.isAllowedShift(state)
        elif(self.actionType(action) == ArcEagerTransitionSystem.REDUCE):
            return self.isAllowedReduce(state)
        elif(self.actionType(action) == ArcEagerTransitionSystem.LEFT_ARC):
            return self.isAllowedLeftArc(state)
        elif(self.actionType(action) == ArcEagerTransitionSystem.RIGHT_ARC):
            return self.isAllowedRightArc(state)
        else:
            assert None

    ## TODO: fix conditdions
    def isAllowedShift(self, state):
        return (not state.endOfInput())

    def isAllowedReduce(self, state):
        return (state.stackSize() > 0)

    def isAllowedLeftArc(self, state):
        # Left-arc requires one or more tokens on the stack but the first token
        # is the root and we do not want a left arc to the root.
        ## FIXME: also requires one in buffer (does endOfInput check that?)

        # do not want left arc to root??
        # ???
        return (state.stackSize() > 1) and (not state.endOfInput())

    def isAllowedRightArc(self, state):
        return (state.stackSize() > 0) and (not state.endOfInput())

    def performAction(self, action, state):
        self.performActionWithoutHistory(action, state)

    def performActionWithoutHistory(self, action, state):
        if self.actionType(action) == ArcEagerTransitionSystem.SHIFT:
            self.performShift(state)
        elif self.actionType(action) == ArcEagerTransitionSystem.REDUCE:
            self.performReduce(state)
        elif self.actionType(action) == ArcEagerTransitionSystem.LEFT_ARC:
            self.performLeftArc(state, self.label(action))
        elif self.actionType(action) == ArcEagerTransitionSystem.RIGHT_ARC:
            self.performRightArc(state, self.label(action))
        else:
            assert(None)

    def performShift(self, state):
        assert self.isAllowedShift(state)
        state.push(state.next())
        state.advance()

    def performReduce(self, state):
        assert self.isAllowedReduce(state)
        state.pop()


    '''
    Arc Standard
    # S|i|j
    # if HEAD(i) == j... add left arc j->i:
    # (S|i|j, B) => (S|j, B)
    def performLeftArc(self, state, label):
        assert self.isAllowedLeftArc(state)
        s_j = state.pop()
        s_i = state.pop()
        state.addArc(s_i, s_j, label)
        state.push(s_j)
    '''

    # Arc Eager
    # (S|i, j|B) => (S, j|B)
    # add left arc j->i:
    def performLeftArc(self, state, label):
        assert self.isAllowedLeftArc(state)
        s_j = state.next()
        s_i = state.pop()
        state.addArc(s_i, s_j, label)

    '''
    Arc Standard
    # S|i|j
    # if HEAD(j) == i... (and we are done with children to the right of j)
    # add right arc i->j:
    # (S|i|j, B) => (S|i, B)
    def performRightArc(self, state, label):
        assert self.isAllowedRightArc(state)
        s_j = state.pop()
        s_i = state.pop()
        state.addArc(s_j, s_i, label)
        state.push(s_i)
    '''

    # Arc Eager
    # (S|i, j|B) => (S|i|j, B)
    # add right arc i->j:
    def performRightArc(self, state, label):
        assert self.isAllowedRightArc(state)
        s_j = state.next()
        s_i = state.pop()
        state.addArc(s_j, s_i, label)
        state.push(s_i)

        # next token
        # add to S
        state.push(s_j)
        # signify that we've pushed the next token onto the stack
        # remove from B
        state.advance()

    #def isDeterministicState(self, state):
    #    return state.stackSize() < 2 and (not state.endOfInput())

    def isFinalState(self, state):
        #return state.endOfInput() and (state.stackSize() < 2)
        #return state.endOfInput() and self.didLastReduce

        if state.endOfInput():
            for i in range(state.numTokens()):
                print(i, state.head(i))
                assert state.head(i) == state.goldHead(i)

        return state.endOfInput()

    def actionAsTuple(self, action):
        if(self.actionType(action) == ArcEagerTransitionSystem.SHIFT):
            return (ArcEagerTransitionSystem.SHIFT,)
        if(self.actionType(action) == ArcEagerTransitionSystem.REDUCE):
            return (ArcEagerTransitionSystem.REDUCE,)
        elif(self.actionType(action) == ArcEagerTransitionSystem.LEFT_ARC):
            return (ArcEagerTransitionSystem.LEFT_ARC, self.label(action))
        elif(self.actionType(action) == ArcEagerTransitionSystem.RIGHT_ARC):
            return (ArcEagerTransitionSystem.RIGHT_ARC, self.label(action))
        else:
            return None

    def actionAsString(self, action, state, feature_maps):
        if(self.actionType(action) == ArcEagerTransitionSystem.SHIFT):
            return 'SHIFT'
        if(self.actionType(action) == ArcEagerTransitionSystem.REDUCE):
            return 'REDUCE'
        elif(self.actionType(action) == ArcEagerTransitionSystem.LEFT_ARC):
            return 'LEFT_ARC(' + \
                feature_maps['label'].indexToValue(self.label(action)) + ')'
        elif(self.actionType(action) == ArcEagerTransitionSystem.RIGHT_ARC):
            return 'RIGHT_ARC(' + \
                feature_maps['label'].indexToValue(self.label(action)) + ')'
        else:
            return 'UNKNOWN'
