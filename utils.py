import tensorflow as tf
from tensorflow.python.ops import control_flow_ops as cf

'''
Replace all digits with 9s like SyntaxNet
'''
def normalizeDigits(form):
    newform = ''
    for i in range(len(form)):
        if ord(form[i]) >= ord('0') and ord(form[i]) <= ord('9'):
            newform += '9'
        else:
            newform += form[i]
    return newform

'''
Gets array shape of dynamically shaped tensors

Ex.
tensorPrintShape(dense_golden, [dense_golden], 'dense_golden shape')
'''
def tensorPrintShape(inp, data, comment):
    def np_print(*args):
        for x in args:
            print(comment, x.shape)
    return cf.with_dependencies([tf.py_func(np_print, data, [])], inp)

'''
Ex.
tensorPrint(dense_golden, [dense_golden], 'dense_golden data')
'''
def tensorPrint(inp, data, comment):
    def np_print(*args):
        for x in args:
            print(comment, x)
    return cf.with_dependencies([tf.py_func(np_print, data, [])], inp)

tensorDumpValsCallCount = {}

'''
Ex.
tensorDumpVals(dense_golden, [dense_golden], '/tmp/ash_dense_golden_1', 1)
'''
# print only the desired_iter'th time the function is called (1-based)
# for this particular filename
def tensorDumpVals(inp, data, fname, desired_iter):
    global tensorDumpValsCallCount

    def np_print(*args):
        global tensorDumpValsCallCount

        if fname not in tensorDumpValsCallCount:
            tensorDumpValsCallCount[fname] = 0
        tensorDumpValsCallCount[fname] += 1

        # only execute for the iteration # desired
        if tensorDumpValsCallCount[fname] == desired_iter:
            fd = open(fname, 'w')

            for x in args:
                for elem in x.flatten():
                    fd.write('%.8f\n' % elem)

            fd.close()

    return cf.with_dependencies([tf.py_func(np_print, data, [])], inp)

'''
Ex.
tensorDumpValsAllIter(dense_golden, [dense_golden], '/tmp/ash_dense_golden')
'''

def tensorDumpValsAllIter(inp, data, fname):
    global tensorDumpValsCallCount

    def np_print(*args):
        global tensorDumpValsCallCount

        if fname not in tensorDumpValsCallCount:
            tensorDumpValsCallCount[fname] = 0
        tensorDumpValsCallCount[fname] += 1

        fd = open('%s_%04d' % (fname, tensorDumpValsCallCount[fname]), 'w')

        for x in args:
            for elem in x.flatten():
                fd.write('%.8f\n' % elem)

        fd.close()

    return cf.with_dependencies([tf.py_func(np_print, data, [])], inp)

'''
See SyntaxNet utils.h
'''
kPunctuation = [
    (33, 35),       (37, 42),       (44, 47),       (58, 59),
    (63, 64),       (91, 93),       (95, 95),       (123, 123),
    (125, 125),     (161, 161),     (171, 171),     (183, 183),
    (187, 187),     (191, 191),     (894, 894),     (903, 903),
    (1370, 1375),   (1417, 1418),   (1470, 1470),   (1472, 1472),
    (1475, 1475),   (1478, 1478),   (1523, 1524),   (1548, 1549),
    (1563, 1563),   (1566, 1567),   (1642, 1645),   (1748, 1748),
    (1792, 1805),   (2404, 2405),   (2416, 2416),   (3572, 3572),
    (3663, 3663),   (3674, 3675),   (3844, 3858),   (3898, 3901),
    (3973, 3973),   (4048, 4049),   (4170, 4175),   (4347, 4347),
    (4961, 4968),   (5741, 5742),   (5787, 5788),   (5867, 5869),
    (5941, 5942),   (6100, 6102),   (6104, 6106),   (6144, 6154),
    (6468, 6469),   (6622, 6623),   (6686, 6687),   (8208, 8231),
    (8240, 8259),   (8261, 8273),   (8275, 8286),   (8317, 8318),
    (8333, 8334),   (9001, 9002),   (9140, 9142),   (10088, 10101),
    (10181, 10182), (10214, 10219), (10627, 10648), (10712, 10715),
    (10748, 10749), (11513, 11516), (11518, 11519), (11776, 11799),
    (11804, 11805), (12289, 12291), (12296, 12305), (12308, 12319),
    (12336, 12336), (12349, 12349), (12448, 12448), (12539, 12539),
    (64830, 64831), (65040, 65049), (65072, 65106), (65108, 65121),
    (65123, 65123), (65128, 65128), (65130, 65131), (65281, 65283),
    (65285, 65290), (65292, 65295), (65306, 65307), (65311, 65312),
    (65339, 65341), (65343, 65343), (65371, 65371), (65373, 65373),
    (65375, 65381), (65792, 65793), (66463, 66463), (68176, 68184)
]

'''
Determines if the specified unicode ordinal is punctuation or not
'''
def isPunctuation(uni_ord):
    assert type(uni_ord) is int
    i = 0
    while kPunctuation[i][0] > 0:
        if uni_ord < kPunctuation[i][0]:
            return False
        if uni_ord <= kPunctuation[i][1]:
            return True
        i += 1
    return False

'''
Returns true if word consists of punctuation characters.
'''
def isPunctuationToken(word):
    for c in word:
        if not isPunctuation(ord(c)):
            return False
    return True

'''
Determine if tag is a punctuation tag.
'''
def isPunctuationTag(tag):
    # match SyntaxNet behavior
    #if len(tag) == 0:
    #    return False
    for c in tag:
        if (c != ',' and c != ':' and c != '.' and c != '\'' and c != '`'):
            return False
    return True

'''
Returns true if tag is non-empty and has only punctuation or parens
symbols.
'''
def isPunctuationTagOrParens(tag):
    if len(tag) == 0:
        return False
    for c in tag:
        if (c != '(' and c != ')' and c != ',' and c != ':' and c != '.' and \
                c != '\'' and c != '`'):
            return False
    return True

# FIXME: empty tags might show as '_' in CoNLL. '_' behavior is not
# well-defined in the specification.

'''
Return whether or not we should score a token based on the current
scoring strategy
'''
def shouldScoreToken(word, tag, scoring_strategy):
    if scoring_strategy == 'default':
        return len(tag) == 0 or not isPunctuationTag(tag)
    elif scoring_strategy == 'conllx':
        return not isPunctuationToken(word)
    elif scoring_strategy == 'ignore_parens':
        return not isPunctuationTagOrParens(tag)
    assert None, 'unknown scoring strategy: ' + scoring_strategy
