# based on parser_features.cc

import logging
from conll_utils import ParsedConllSentence, ParsedConllToken
from parser_state import ParserState

GlobalFeatureStringCache = dict()

logger = logging.getLogger('FeatureExtractor')

'''
Represents a feature (input.tag, stack.child(-1).sibling(1).label, etc)

Currently only tag, word, and label are possible
'''
class FeatureType(object):
    #FEATURE_TAG     =   0
    #FEATURE_LABEL   =   1

    KNOWN_FEATURE_TYPES = ['tag', 'label', 'word']

    def __init__(self, feature_major_type, feature_name):
        assert feature_major_type in FeatureType.KNOWN_FEATURE_TYPES, \
            'unsupported feature major type ' + str(feature_major_type)

        # 'label', etc
        self.major_type = feature_major_type

        # 'stack.child(-1).sibling(1).label', etc
        self.name = feature_name

'''
A settable focus integer (allows us to pass by reference)
'''
class Focus(object):
    def __init__(self, focus):
        self.val = focus

'''
Locator that takes a parser state and produces a focus integer
(a token index indexable via getToken())
'''
class ParserLocator(object):
    def __init__(self, parser):
        self.parser = parser

    def getFocus(self):
        assert None, 'getFocus() must be overridden'

'''
args[0]: optional: n index of input(n)
         if not specified, index 0 is looked up
'''
class InputParserLocator(ParserLocator):
    def __init__(self, parser, args):
        super().__init__(parser)
        assert len(args) == 0 or len(args) == 1
        if len(args) == 0:
            args.append(0)
        assert type(args[0]) is int
        self.args = args

    def getFocus(self):
        return Focus(self.parser.input(self.args[0]))
    
'''
args[0]: optional: n index of stack(n)
         if not specified, index 0 is looked up
'''
class StackParserLocator(ParserLocator):
    def __init__(self, parser, args):
        super().__init__(parser)
        assert len(args) == 0 or len(args) == 1
        if len(args) == 0:
            args.append(0)
        assert type(args[0]) is int
        self.args = args

    def getFocus(self):
        return Focus(self.parser.stack(self.args[0]))

'''
(ParserState, int) is taken as argument to reference a token
and each inherited index locator looks up a certain sibling/HEAD/etc of this
token
'''
class ParserIndexLocator(object):
    def __init__(self, parser):
        #assert type(tokenIndex) is int
        # -2 could signify no existing element, -1 signifies HEAD
        #assert tokenIndex >= -2
        #assert tokenIndex < parser.numTokens()
        self.parser = parser
        #self.tokenIndex = tokenIndex

    def updateArgs(self, focus):
        assert None, 'updateArgs() must be overridden'

'''
Arguments: args[0]: number of times to call head() function
'''
class HeadFeatureLocator(ParserIndexLocator):
    def __init__(self, parser, args):
        super().__init__(parser)
        assert len(args) == 1
        assert type(args[0]) is int
        self.args = args

    def updateArgs(self, focus):
        levels = self.args[0]
        assert levels >= 1

        # same logic as SyntaxNet
        if (focus.val < -1) or (focus.val >= self.parser.numTokens()):
            focus.val = -2
            return
        focus.val = self.parser.parent(focus.val, levels)


'''
Arguments: args[0]: get n'th child (< 0 indicates leftmost, > 0 indicates
rightmost)
'''
class ChildFeatureLocator(ParserIndexLocator):
    def __init__(self, parser, args, doLogging=True):
        super().__init__(parser)
        assert len(args) == 1
        assert type(args[0]) is int
        self.args = args
        self.doLogging = doLogging

    def updateArgs(self, focus):
        levels = self.args[0]
        assert levels != 0

        # same logic as SyntaxNet
        if (focus.val < -1) or (focus.val >= self.parser.numTokens()):
            if self.doLogging:
                self.logger.debug('ChildFeatureLocator: focus=-2')
            focus.val = -2
            return

        oldfocus = focus.val
        if (levels < 0):
            focus.val = self.parser.leftmostChild(focus.val, -levels)
            if self.doLogging:
                self.logger.debug('ChildFeatureLocator: leftmostChild: levels=%d, '
                    'focus=%d->%d' % (levels, oldfocus, focus.val))
        else:
            focus.val = self.parser.rightmostChild(focus.val, levels)
            if self.doLogging:
                self.logger.debug('ChildFeatureLocator: rightmostChild: levels=%d, '
                    'focus=%d->%d' % (levels, oldfocus, focus.val))

'''
Arguments: args[0]: get n'th sibling (< 0 indicates to left, > 0 indicates to
right)
'''
class SiblingFeatureLocator(ParserIndexLocator):
    def __init__(self, parser, args, doLogging=True):
        super().__init__(parser)
        assert len(args) == 1
        assert type(args[0]) is int
        self.args = args
        self.doLogging = doLogging

    def updateArgs(self, focus):
        position = self.args[0]
        assert position != 0

        # same logic as SyntaxNet
        if (focus.val < -1) or (focus.val >= self.parser.numTokens()):
            if self.doLogging:
                self.logger.debug('SiblingFeatureLocator: focus=-2')
            focus.val = -2
            return

        oldfocus = focus.val
        if (position < 0):
            focus.val = self.parser.leftSibling(focus.val, -position)
            if self.doLogging:
                self.logger.debug('SiblingFeatureLocator: leftSibling: ' \
                    'position=%d, ' \
                    'focus=%d->%d' % (position, oldfocus, focus.val))
        else:
            focus.val = self.parser.rightSibling(focus.val, position)
            if self.doLogging:
                self.logger.debug('SiblingFeatureLocator: rightSibling: ' \
                    'position=%d, ' \
                    'focus=%d->%d' % (position, oldfocus, focus.val))

'''
Decodes a feature separated between a dot value into a feature name and argument
list

Only supports integer arguments for now

Input:
 FeatureString: label

Output:
 FeatureName: label
 FeatureArgs: []
--
Input:
 FeatureString: input(0)

Output:
 FeatureName: input
 FeatureArgs: [0]
--
Input:
 FeatureString: xxx(0,5)

Output:
 FeatureName: xxx
 FeatureArgs: [0, 5]
'''
def decodeFeatureString(featureString):
    if '(' in featureString:
        featureName = featureString.split('(')[0]
        tmp = featureString.split('(')[1].split(')')[0].split(',')
        featureArgs = []

        for t in tmp:
            t = t.strip()
            assert t.lstrip('-').isdecimal()
            featureArgs.append(int(t))
        
        return featureName, featureArgs
    else:
        return featureString, []

'''
Represents all feature groups' values retrieved for one parser state at once
'''
class FeatureVector(object):
    def __init__(self):
        self.types = []
        self.values = []

    '''
    Returns values concatenated for similar feature major types
    (like feature groups per-token)

    However in reality we usually concatenate features with each other at a
    batch-level, not at a token-level
    '''
    def concatenateSimilarTypes(self):
        all_major_types = set()
        for t in self.types:
            all_major_types.add(t.major_type)

        all_major_types = list(all_major_types)
        # for consistency
        all_major_types.sort()

        concat_major_types = []
        concat_values = []

        for t in all_major_types:
            concat_major_types.append(t)
            concat_values.append([])
            for i in range(len(self.types)):
                if self.types[i].major_type == t:
                    concat_values[-1].append(self.values[i])
        return concat_major_types, concat_values

'''
Given feature strings, returns FeatureVector for a particular parser state
'''
class SparseFeatureExtractor(object):
    def __init__(self, feature_strings, feature_maps):
        self.feature_strings = feature_strings
        self.feature_maps = feature_maps
        self.logger = logging.getLogger('SparseFeatureExtractor')

    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['logger']          # remove logger entry
        return odict

    def __setstate__(self, dict):
        self.logger = logging.getLogger('SparseFeatureExtractor')
        self.__dict__.update(dict)   # update attributes

    '''
    doLogging=False: don't log if we're just in init mode where we determine
                     major feature types during initialization, etc...
    '''
    def extract(self, parser, doLogging=True):
        fvec = FeatureVector()
        for fstr in self.feature_strings:
            ftype, fval = self.extractOne(parser, fstr, doLogging=doLogging)
            fvec.types.append(ftype)
            fvec.values.append(fval)
        return fvec

    def extractOne(self, parser, featureString, doLogging=True):
        global GlobalFeatureStringCache
        
        if featureString not in GlobalFeatureStringCache:
            featureParts = featureString.split('.')

            # must reference at least one focus and at least one feature
            # (tag/label/etc), therefore, at least two elements
            assert len(featureParts) >= 2

            # featureParts: ['stack(1)', 'child(-1)', 'sibling(1)', 'word']
            decodedParts = []
            for p in featureParts:
                p = p.strip()
                decodedParts.append(decodeFeatureString(p))
            # decodedParts: [('stack', [1]), ('child', [-1]), ('sibling', [1]),
            #                ('word', [])]
            assert len(decodedParts) >= 2
            assert(decodedParts[0][0] == 'input' or decodedParts[0][0] == 'stack')

            GlobalFeatureStringCache[featureString] = decodedParts
        else:
            decodedParts = GlobalFeatureStringCache[featureString]

        # start setting focus and follow focus modifiers until real feature
        # (tag/label/etc)
        focus = None
        feature_name = featureString
        feature_major_type = None #featureString.split('.')[-1]
        feature_index = None

        for d in decodedParts:
            fname = d[0]
            fargs = d[1]
            if fname == 'input':
                assert feature_index == None, \
                    'can\'t update focus if feature is already set'
                if len(fargs) == 0:
                    fargs=[0]
                focus = parser.input(fargs[0])
            elif fname == 'stack':
                assert feature_index == None, \
                    'can\'t update focus if feature is already set'
                if len(fargs) == 0:
                    fargs=[0]
                focus = parser.stack(fargs[0])
            elif fname == 'head':
                assert focus != None, 'can\'t take HEAD of null focus'
                assert feature_index == None, \
                    'can\'t update focus if feature is already set'

                assert len(fargs) == 1
                levels = fargs[0]
                assert levels >= 1

                # same logic as SyntaxNet
                if (focus < -1) or (focus >= parser.numTokens()):
                    focus = -2
                else:
                    focus = parser.parent(focus, levels)
            elif fname == 'child':
                assert focus != None, 'can\'t take CHILD of null focus'
                assert feature_index == None, \
                    'can\'t update focus if feature is already set'

                levels = fargs[0]
                assert levels != 0

                # same logic as SyntaxNet
                if (focus < -1) or (focus >= parser.numTokens()):
                    if doLogging:
                        self.logger.debug('ChildFeatureLocator: focus=-2')
                    focus = -2
                else:
                    oldfocus = focus
                    if (levels < 0):
                        focus = parser.leftmostChild(focus, -levels)
                        if doLogging:
                            self.logger.debug('ChildFeatureLocator: leftmostChild: levels=%d, '
                                'focus=%d->%d' % (levels, oldfocus, focus))
                    else:
                        focus = parser.rightmostChild(focus, levels)
                        if doLogging:
                            self.logger.debug('ChildFeatureLocator: rightmostChild: levels=%d, '
                                'focus=%d->%d' % (levels, oldfocus, focus))

                #ChildFeatureLocator(parser, fargs, \
                #    doLogging=doLogging).updateArgs(focus)
            elif fname == 'sibling':
                assert focus != None, 'can\'t take SIBLING of null focus'
                assert feature_index == None, \
                    'can\'t update focus if feature is already set'
                position = fargs[0]
                assert position != 0

                # same logic as SyntaxNet
                if (focus < -1) or (focus >= parser.numTokens()):
                    if doLogging:
                        self.logger.debug('SiblingFeatureLocator: focus=-2')
                    focus = -2
                else:
                    oldfocus = focus
                    if (position < 0):
                        focus = parser.leftSibling(focus, -position)
                        if doLogging:
                            self.logger.debug('SiblingFeatureLocator: leftSibling: ' \
                                'position=%d, ' \
                                'focus=%d->%d' % (position, oldfocus, focus))
                    else:
                        focus = parser.rightSibling(focus, position)
                        if doLogging:
                            self.logger.debug('SiblingFeatureLocator: rightSibling: ' \
                                'position=%d, ' \
                                'focus=%d->%d' % (position, oldfocus, focus))
            else:
                assert focus != None, 'can\'t request feature of null focus'
                assert feature_index == None, \
                    'can\'t request feature when feature is already set; ' \
                        'nested features not supported'

                if doLogging:
                    self.logger.debug('focus: %d' % focus)

                if fname == 'label':
                    feature_major_type = 'label'
                    if focus == -1:
                        feature_index = \
                            self.feature_maps[feature_major_type] \
                                .valueToIndex('<ROOT>')

                        if doLogging:
                            self.logger.debug('%s: %d (%s)' % \
                                (feature_name, feature_index, '<ROOT>'))

                    elif focus < -1 or focus >= parser.numTokens():
                        feature_index = \
                            self.feature_maps[feature_major_type] \
                                .valueToIndex('<OUTSIDE>')

                        if doLogging:
                            self.logger.debug('%s: %d (%s)' % \
                                (feature_name, feature_index, '<OUTSIDE>'))

                    else:
                        # pulls label from parser itself, which means it won't
                        # be gold as long as parser wasn't initialized with
                        # gold labels
                        feature_index = parser.label(focus)

                        if feature_index == -1:
                            feature_index = \
                                self.feature_maps[feature_major_type] \
                                    .valueToIndex('<ROOT>')

                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index, '<ROOT>'))
                        else:
                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index, \
                                    self.feature_maps[feature_major_type] \
                                        .indexToValue(parser.label(focus))))

                elif fname == 'word':
                    feature_major_type = 'word'
                    if focus < 0 or focus >= parser.numTokens():
                        feature_index = \
                            self.feature_maps[feature_major_type] \
                                .valueToIndex('<OUTSIDE>')

                        if doLogging:
                            self.logger.debug('%s: %d (%s)' % \
                                (feature_name, feature_index, '<OUTSIDE>'))
                    else:
                        try:
                            feature_index = \
                                self.feature_maps[feature_major_type] \
                                    .valueToIndex(parser.getToken(
                                        focus).FORM)

                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index, \
                                    parser.getToken(focus).FORM))
                        except: # Out of Vocabulary
                            feature_index = \
                                self.feature_maps[feature_major_type] \
                                    .valueToIndex('<UNKNOWN>')

                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index, '<UNKNOWN>'))
                elif fname == 'tag':
                    feature_major_type = 'tag'
                    if focus < 0 or focus >= parser.numTokens():
                        feature_index = self.feature_maps[feature_major_type] \
                            .valueToIndex('<OUTSIDE>')

                        if doLogging:
                            self.logger.debug('%s: %d (%s)' % (feature_name, \
                                feature_index, '<OUTSIDE>'))
                    else:
                        try:
                            feature_index = \
                                self.feature_maps[feature_major_type] \
                                    .valueToIndex(parser.getToken(
                                        focus).XPOSTAG)

                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index,
                                    parser.getToken(focus).XPOSTAG))
                        except: # Out of Vocabulary
                            feature_index = \
                                self.feature_maps[feature_major_type] \
                                    .valueToIndex('<UNKNOWN>')

                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index, '<UNKNOWN>'))
                else:
                    assert None, 'unknown feature name \'' + fname + '\''

        assert feature_name != None, 'feature name undetermined'
        assert feature_major_type != None, 'feature major type undetermined'
        assert feature_index != None, 'focus set but feature never requested'
        return FeatureType(feature_major_type, feature_name), feature_index

    '''
    featureString: stack(1).child(-1).sibling(1).word
    doLogging=False: don't log if we're just in init mode
    '''
    def extractOneOld(self, parser, featureString, doLogging=True):
        featureParts = featureString.split('.')

        # must reference at least one focus and at least one feature
        # (tag/label/etc), therefore, at least two elements
        assert len(featureParts) >= 2

        # featureParts: ['stack(1)', 'child(-1)', 'sibling(1)', 'word']
        decodedParts = []
        for p in featureParts:
            p = p.strip()
            decodedParts.append(decodeFeatureString(p))
        # decodedParts: [('stack', [1]), ('child', [-1]), ('sibling', [1]),
        #                ('word', [])]
        assert len(decodedParts) >= 2
        assert(decodedParts[0][0] == 'input' or decodedParts[0][0] == 'stack')

        # start setting focus and follow focus modifiers until real feature
        # (tag/label/etc)
        focus = None
        feature_name = featureString
        feature_major_type = None
        feature_index = None

        for d in decodedParts:
            fname = d[0]
            fargs = d[1]
            if fname == 'input':
                assert feature_index == None, \
                    'can\'t update focus if feature is already set'
                focus = InputParserLocator(parser, fargs).getFocus()
            elif fname == 'stack':
                assert feature_index == None, \
                    'can\'t update focus if feature is already set'
                focus = StackParserLocator(parser, fargs).getFocus()
            elif fname == 'head':
                assert focus != None, 'can\'t take HEAD of null focus'
                assert feature_index == None, \
                    'can\'t update focus if feature is already set'
                HeadFeatureLocator(parser, fargs).updateArgs(focus)
            elif fname == 'child':
                assert focus != None, 'can\'t take CHILD of null focus'
                assert feature_index == None, \
                    'can\'t update focus if feature is already set'
                ChildFeatureLocator(parser, fargs, \
                    doLogging=doLogging).updateArgs(focus)
            elif fname == 'sibling':
                assert focus != None, 'can\'t take SIBLING of null focus'
                assert feature_index == None, \
                    'can\'t update focus if feature is already set'
                SiblingFeatureLocator(parser, fargs, \
                    doLogging=doLogging).updateArgs(focus)
            else:
                assert focus != None, 'can\'t request feature of null focus'
                assert feature_index == None, \
                    'can\'t request feature when feature is already set; ' \
                        'nested features not supported'

                if doLogging:
                    self.logger.debug('focus.val: %d' % focus.val)

                if fname == 'label':
                    feature_major_type = 'label'
                    if focus.val == -1:
                        feature_index = \
                            self.feature_maps[feature_major_type] \
                                .valueToIndex('<ROOT>')

                        if doLogging:
                            self.logger.debug('%s: %d (%s)' % \
                                (feature_name, feature_index, '<ROOT>'))

                    elif focus.val < -1 or focus.val >= parser.numTokens():
                        feature_index = \
                            self.feature_maps[feature_major_type] \
                                .valueToIndex('<OUTSIDE>')

                        if doLogging:
                            self.logger.debug('%s: %d (%s)' % \
                                (feature_name, feature_index, '<OUTSIDE>'))

                    else:
                        # pulls label from parser itself, which means it won't
                        # be gold as long as parser wasn't initialized with
                        # gold labels
                        feature_index = parser.label(focus.val)

                        if feature_index == -1:
                            feature_index = \
                                self.feature_maps[feature_major_type] \
                                    .valueToIndex('<ROOT>')

                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index, '<ROOT>'))
                        else:
                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index, \
                                    self.feature_maps[feature_major_type] \
                                        .indexToValue(parser.label(focus.val))))

                elif fname == 'word':
                    feature_major_type = 'word'
                    if focus.val < 0 or focus.val >= parser.numTokens():
                        feature_index = \
                            self.feature_maps[feature_major_type] \
                                .valueToIndex('<OUTSIDE>')

                        if doLogging:
                            self.logger.debug('%s: %d (%s)' % \
                                (feature_name, feature_index, '<OUTSIDE>'))
                    else:
                        try:
                            feature_index = \
                                self.feature_maps[feature_major_type] \
                                    .valueToIndex(parser.getToken(
                                        focus.val).FORM)

                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index, \
                                    parser.getToken(focus.val).FORM))
                        except: # Out of Vocabulary
                            feature_index = \
                                self.feature_maps[feature_major_type] \
                                    .valueToIndex('<UNKNOWN>')

                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index, '<UNKNOWN>'))
                elif fname == 'tag':
                    feature_major_type = 'tag'
                    if focus.val < 0 or focus.val >= parser.numTokens():
                        feature_index = self.feature_maps[feature_major_type] \
                            .valueToIndex('<OUTSIDE>')

                        if doLogging:
                            self.logger.debug('%s: %d (%s)' % (feature_name, \
                                feature_index, '<OUTSIDE>'))
                    else:
                        try:
                            feature_index = \
                                self.feature_maps[feature_major_type] \
                                    .valueToIndex(parser.getToken(
                                        focus.val).XPOSTAG)

                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index,
                                    parser.getToken(focus.val).XPOSTAG))
                        except: # Out of Vocabulary
                            feature_index = \
                                self.feature_maps[feature_major_type] \
                                    .valueToIndex('<UNKNOWN>')

                            if doLogging:
                                self.logger.debug('%s: %d (%s)' % \
                                    (feature_name, feature_index, '<UNKNOWN>'))
                else:
                    assert None, 'unknown feature name \'' + fname + '\''

        assert feature_name != None, 'feature name undetermined'
        assert feature_major_type != None, 'feature major type undetermined'
        assert feature_index != None, 'focus set but feature never requested'
        return FeatureType(feature_major_type, feature_name), feature_index
