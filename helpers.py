import string
import numpy as np
from jamo import h2j, j2hcj, jamo_to_hcj, is_jamo

# we only support modern Jamo
JAMO_LEADS_MODERN = [chr(_) for _ in range(0x1100, 0x1113)]
JAMO_VOWELS_MODERN = [chr(_) for _ in range(0x1161, 0x1176)]
JAMO_TAILS_MODERN = [chr(_) for _ in range(0x11A8, 0x11C3)]

#JAMO_LEADS_HCJ = [list(jamo_to_hcj(_))[0] for _ in JAMO_LEADS_MODERN]
#JAMO_VOWELS_HCJ = [list(jamo_to_hcj(_))[0] for _ in JAMO_VOWELS_MODERN]
#JAMO_TAILS_HCJ = [list(jamo_to_hcj(_))[0] for _ in JAMO_TAILS_MODERN]
# HCJ seem to map to same char, so don't use them.

#JAMO_LEADS_ENCODE = JAMO_LEADS_HCJ + ['FOREIGN', 'NUMBER', 'PUNCT', 'OTHER']
#JAMO_VOWELS_ENCODE = JAMO_VOWELS_HCJ + ['FOREIGN', 'NUMBER', 'PUNCT', 'OTHER']
#JAMO_TAILS_ENCODE = JAMO_TAILS_HCJ + ['FOREIGN', 'NUMBER', 'PUNCT', 'OTHER']

JAMO_ENCODE = JAMO_LEADS_MODERN + JAMO_VOWELS_MODERN + JAMO_TAILS_MODERN + ['FOREIGN', 'NUMBER', 'PUNCT', 'OTHER']

# make sure all items are unique
assert len(set(JAMO_ENCODE)) == len(JAMO_ENCODE), ('%d!=%d' % (len(set(JAMO_ENCODE)), len(JAMO_ENCODE)))

def getJamoEncodingType(j):
    if is_jamo(j):
        return j
    else:
        if j.isdigit():
            return 'NUMBER'
        elif j in string.punctuation:
            return 'PUNCT'
        elif j.isalpha(): # seems to support foreign alphabet
            return 'FOREIGN'
        else:
            return 'OTHER'

'''
Also supports foreign characters
>>> j2hcj(h2j("자모=字母=jamo"))
'ㅈㅏㅁㅗ=字母=jamo'

Outputs Hangul Conjoining Jamo (not Compatibility Jamo)
This is because Conjoining Jamo have all unique lead, vowel, tail indices, convenient for one-hot encoding
'''
def toJamo(w):
    return h2j(w)

'''
Returns an array representing one-hot encoding, with only activateIndex set to 1
'''
def oneHotArray(size, activateIndex):
    ary = []
    for i in range(size):
        ary.append(0)
    ary[activateIndex] = 1
    return ary

'''
Gets the index of the hot element in the array
'''
def getHotElementIndex(ary):
    return np.argmax(ary)

'''
One-hot encodes tag within tag set based on index

Supports encoding null values as first element within one-hot array
'''
def oneHotEncodeWithNull(tag, tagSet):
    assert type(tagSet) is list
    if tag != None:
        assert tag in tagSet
    if tag != None:
        # first element in one-hot is reserved as None
        return oneHotArray(len(tagSet)+1, tagSet.index(tag)+1)
    else:
        return oneHotArray(len(tagSet)+1, 0)

'''
One-hot encodes tag within tag set based on index
'''
def oneHotEncode(tag, tagSet):
    assert type(tagSet) is list
    assert tag in tagSet
    return oneHotArray(len(tagSet), tagSet.index(tag))

'''
One-hot decodes tag within tag set based on index
'''
def oneHotDecode(oneHotArray, tagSet):
    assert type(tagSet) is list
    assert len(oneHotArray) == len(tagSet)
    return tagSet[getHotElementIndex(oneHotArray)]

'''
One-hot decodes tag within tag set based on index

Supports decoding null values from first element within one-hot array
'''
def oneHotDecodeWithNull(oneHotArray, tagSet):
    assert type(tagSet) is list
    assert len(oneHotArray) == len(tagSet)+1 # for null
    hotIndex = getHotElementIndex(oneHotArray)
    if hotIndex == 0:
        return None
    else:
        return tagSet[hotIndex-1]
