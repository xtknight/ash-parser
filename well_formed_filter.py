'''
Well-Formed Filter
(document_filters.cc)

Check that input is single-root, connected, acyclic, and projective.
'''
import logging
logger = logging.getLogger('WellFormedFilter')
from projectivize_filter import checkProjectivity

'''
Determine whether all HEADs are within the bounds of the sentence
'''
def allHeadsExist(sentence):
    minIndex = -1 # root token
    maxIndex = len(sentence.tokens)-1

    for t in sentence.tokens:
        if t.HEAD < minIndex or t.HEAD > maxIndex:
            return False

    return True

'''
Determine whether the sentence is single rooted
'''
def isSingleRooted(sentence):
    allHeads = []
    for t in sentence.tokens:
        if t.HEAD == -1:
            allHeads.append(t)
    return len(allHeads) == 1

'''
Determine whether or not the sentence has a cycle (in HEADs)
'''
def hasCycle(sentence):
    visited = [-1 for t in sentence.tokens]

    for i in range(len(sentence.tokens)):
        # Already visited node
        if visited[i] != -1:
            continue

        t = i
        while t != -1:
            if visited[t] == -1:
                # If it is not visited yet, mark it.
                visited[t] = i
            elif visited[t] < i:
                # If the index number is smaller than index and not -1, the
                # token has already been visited.
                break
            else:
                # Loop detected
                return True
            t = sentence.tokens[t].HEAD

    return False

class WellFormedFilter(object):
    def __init__(self):
        self.nonProjectiveCount = 0
        self.projectivizedCount = 0
        self.nonWellFormedCount = 0

    '''
    Determine whether the sentence can be parsed by arc-standard and arc-eager
    or not

    projectivize: whether to make non-projective sentences projective
    '''
    def isWellFormed(self, sentence, projectivize=False):
        if len(sentence.tokens) == 0:
            logger.debug('Not well-formed: token length is zero')
            logger.debug('"'+sentence.toSimpleRepresentation()+'"')
            self.nonWellFormedCount += 1
            return False

        if not allHeadsExist(sentence):
            logger.debug('Not well-formed: not all HEADs exist as tokens')
            logger.debug('"'+sentence.toSimpleRepresentation()+'"')
            self.nonWellFormedCount += 1
            return False

        if not isSingleRooted(sentence):
            logger.debug('Not well-formed: tree doesn\'t have single ROOT')
            logger.debug('"'+sentence.toSimpleRepresentation()+'"')
            self.nonWellFormedCount += 1
            return False

        if hasCycle(sentence):
            logger.debug('Not well-formed: tree has a cycle')
            logger.debug('"'+sentence.toSimpleRepresentation()+'"')
            self.nonWellFormedCount += 1
            return False

        if not checkProjectivity(sentence, projectivize=projectivize):
            self.nonProjectiveCount += 1

            # if it wasn't projective
            if not projectivize:
                # ... and we didn't projectivize it... then it's invalid
                logger.debug('Not well-formed: non-projective and' \
                    ' projectivize disabled')
                logger.debug('"'+sentence.toSimpleRepresentation()+'"')

                # only count them as non-well-formed when projectivize is off
                self.nonWellFormedCount += 1
                return False
            else:
                # we succesfully projectivized a non-projective sentence
                # consider well-formed
                self.projectivizedCount += 1

        # if we did projectivize it, then we can keep going

        return True
