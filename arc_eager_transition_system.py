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

    def getDepRelation(self, idx_parent, idx_child, state):
        if idx_child == -1:
            return None # root word
        assert idx_child >= 0
        
        if state.goldHead(idx_child) == idx_parent:
            # fixme: if label is -1 then??
            return state.goldLabel(idx_child)
        else:
            return None

    def getNextGoldAction(self, state):
        b0 = state.input(0)

        bInput = -2
        bInc = 0
        bItems = []
        while bInput != -2:
            bInput = state.input(bInc)
            if bInput == -2:
                break
            bItems.append(bInput)
            bInc += 1

        #print('B:', bItems)

        if state.stackSize() > 0:
            #print('S:', state.stack_)

            s0 = state.stack(0)
            rel = self.getDepRelation(b0, s0, state)
            if rel is not None:
                #print('return L-A action', self.leftArcAction(rel))
                return self.leftArcAction(rel)

            rel = self.getDepRelation(s0, b0, state)
            if rel is not None:
                #print('return R-A action', self.rightArcAction(rel))
                return self.rightArcAction(rel)

            flag = False
            for k in range(-1, s0): # s0 goes as low as -1 unlike NLTK
                if self.getDepRelation(k, b0, state) is not None:
                    flag = True
                if self.getDepRelation(b0, k, state) is not None:
                    flag = True

            if flag:
                #print('return R action', self.reduceAction())
                return self.reduceAction()

        # S|i
        # nothing else we can do except shift to the next required arc
        # (S, i|B) => (S|i, B)
        #print('return S action', self.shiftAction())
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

    def isAllowedShift(self, state):
        return (not state.endOfInput())

    def isAllowedReduce(self, state):
        if state.stackSize() == 0:
            return False

        idx_wi = state.stack(0)
        flag = False
        for (idx_parent, r, idx_child) in state.arcs_:
            if idx_child == idx_wi:
                flag = True
        if not flag:
            return False

        return True

    def isAllowedLeftArc(self, state):
        if state.endOfInput() or state.stackSize() == 0:
            return False

        # this is the root element
        if state.input(0) == -1:
            return False

        # in nltk code, 0 if root node
        # here, -1 if root node
        idx_wi = state.stack(0)
        flag = True

        # but the problem is, root node is not in head_
        # or label_
        # (they start from node 0, not -1)
        # but head and label are initialized to -1 always,
        # so it's confusing.
        # store arcs_ separately based on transitions made
        # to state
        for (idx_parent, r, idx_child) in state.arcs_:
            if idx_child == idx_wi:
                flag = False
        if not flag:
            return False
            
        return True

    def isAllowedRightArc(self, state):
        if state.endOfInput() or state.stackSize() == 0:
            return False

        return True

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

    # Arc Eager
    # (S|i, j|B) => (S, j|B)
    # add left arc j->i:
    def performLeftArc(self, state, label):
        assert self.isAllowedLeftArc(state)
        s_j = state.next()
        s_i = state.pop()
        state.addArc(s_i, s_j, label)

    # Arc Eager
    # (S|i, j|B) => (S|i|j, B)
    # add right arc i->j:
    def performRightArc(self, state, label):
        assert self.isAllowedRightArc(state)
        s_j = state.next()
        s_i = state.stack(0)
        state.addArc(s_j, s_i, label)

        # next token
        # add to S
        state.push(s_j)
        # signify that we've pushed the next token onto the stack
        # remove from B
        state.advance()

    def isFinalState(self, state):
        if state.endOfInput():
            for i in range(state.numTokens()):
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
