'''
Projectivize Filter
(document_filters.cc)

Check whether the given sentence is projective or not.

Return value is whether the sentence was originally projective or not.
  - If it wasn't, return False
  - If it was, return True

projectivize parameter: whether or not to fix sentence to be projective
(does not affect return value)
'''

import copy
import logging
logger = logging.getLogger('ProjectivizeFilter')

def checkProjectivity(sentence, projectivize=False):
    if projectivize:
        oldsentence = copy.deepcopy(sentence)

    wasProjective = True
    num_tokens = len(sentence.tokens)

    # Left and right boundaries for arcs. The left and right ends of an arc are
    # bounded by the arcs that pass over it. If an arc exceeds these bounds it
    # will cross an arc passing over it, making it a non-projective arc.

    left = [None for i in range(num_tokens)]
    right = [None for i in range(num_tokens)]

    # Lift the shortest non-projective arc until the document is projective.
    while True:
        # Initialize boundaries to the whole document for all arcs.
        for i in range(num_tokens):
            left[i] = -1
            right[i] = num_tokens - 1

        # Find left and right bounds for each token.
        for i in range(num_tokens):
            head_index = sentence.tokens[i].HEAD

            # Find left and right end of arc
            l = min(i, head_index)
            r = max(i, head_index)

            # Bound all tokens under the arc.
            for j in range(l+1, r):
                if left[j] < l:
                    left[j] = l
                if right[j] > r:
                    right[j] = r

        # Find deepest non-projective arc.
        deepest_arc = -1
        max_depth = -1

        # The non-projective arcs are those that exceed their bounds.
        for i in range(num_tokens):
            head_index = sentence.tokens[i].HEAD

            if head_index == -1:
                # any crossing arc must be deeper
                continue

            l = min(i, head_index)
            r = max(i, head_index)

            left_bound = max(left[l], left[r])
            right_bound = min(right[l], right[r])

            if (l < left_bound) or (r > right_bound):
                # Found non-projective arc.
                logger.debug('Found non-projective arc')
                wasProjective = False
                if not projectivize:
                    return wasProjective

                # Pick the deepest as the best candidate for lifting.
                depth = 0
                j = i
                while j != -1:
                    depth += 1
                    j = sentence.tokens[j].HEAD
                
                if depth > max_depth:
                    deepest_arc = i
                    max_depth = depth

        # If there are no more non-projective arcs we are done.
        if deepest_arc == -1:
            if not wasProjective:
                logger.debug('Projectivized non-projective arc')
                logger.debug('Before\n' + oldsentence.toFileOutput())
                logger.debug('After\n' + sentence.toFileOutput())
            return wasProjective

        # Lift non-projective arc.
        lifted_head = sentence.tokens[sentence.tokens[deepest_arc].HEAD].HEAD
        sentence.tokens[deepest_arc].HEAD = lifted_head

    assert None
