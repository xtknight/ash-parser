'''
Represents a feature encoded as a sparse index
'''
class UnsortedIndexEncodedFeatureMap(object):
    def __init__(self):
        # base value doesn't exist
        # (don't set to 0, as 0 should be a valid index in that case)
        self.lastBaseValue = -1
        self.isFinalized = False
        self.indexToValueMap = dict()
        self.valueToIndexMap = dict()

    '''
    Increment frequency for the specified term
    If we've never seen this term before, make an entry for it
    '''
    def addTerm(self, term):
        assert not self.isFinalized

        self.lastBaseValue += 1
        self.indexToValueMap[self.lastBaseValue] = term
        self.valueToIndexMap[term] = self.lastBaseValue
    
    '''
    Finalize indices and sort by descending frequency of each term, and then
    alphabetically
    Index 0 will be the most frequent term
    '''
    def finalizeBaseValues(self):
        assert not self.isFinalized
        assert len(self.valueToIndexMap) == len(self.indexToValueMap), \
            'index<->value map length mismatch'
        self.isFinalized = True

    '''
    Append special value after finalization, like <NULL>, etc...
    '''
    def appendSpecialValue(self, term):
        assert self.isFinalized
        if term in self.valueToIndexMap:
            return # no need to add another index for it

        newTermIndex = len(self.valueToIndexMap)
        assert newTermIndex not in self.indexToValueMap
        self.indexToValueMap[newTermIndex] = term
        self.valueToIndexMap[term] = newTermIndex

        assert len(self.valueToIndexMap) == len(self.indexToValueMap)

    def valueToIndex(self, v):
        assert self.isFinalized
        return self.valueToIndexMap[v]

    def indexToValue(self, i):
        assert self.isFinalized
        return self.indexToValueMap[i]

    '''
    Get the number of possible unique values for this feature
    (optionally excluding special features)
    '''
    def getDomainSize(self, includeSpecial=True):
        assert self.isFinalized
        if includeSpecial:
            return len(self.valueToIndexMap)
        else:
            return self.lastBaseValue + 1

'''
Represents a feature encoded as a sparse index

Sorts base values by frequency in descending order and then name in ascending
order

Sorting ensures equivalent behavior per run
'''
class IndexEncodedFeatureMap(UnsortedIndexEncodedFeatureMap):
    def __init__(self):
        super().__init__()
        self.freq = dict()

    def addTerm(self, term):
        assert None, 'addTerm() not allowed in IndexEncodedFeatureMap'

    '''
    Increment frequency for the specified term
    If we've never seen this term before, make an entry for it
    '''
    def incrementTerm(self, term):
        assert not self.isFinalized
        if term not in self.freq:
            self.freq[term] = 0
        self.freq[term] += 1

    def loadFrom(self, fname):
        assert not self.isFinalized
        fd = open(fname, 'r', encoding='utf-8')
        contents = fd.read()
        fd.close()

        ln_num = 0
        itemCount = 0
        currentItem = 0
        for ln in contents:
            ln = ln.strip()
            if not ln:
                continue
            ln_num += 1
            if ln_num == 1:
                itemCount = int(ln)
            else:
                assert ln.count(' ') == 2

                term, freq = ln.split()
                term = term.strip()
                freq = int(freq)

                assert term not in self.freq, 'term already loaded'

                self.freq[term] = freq

                currentItem += 1
                if currentItem >= itemCount:
                    break

        assert currentItem == itemCount, 'not all items loaded properly'

        # caller should do finalization
        # sets finalize flag and counts base values
        # self.finalizeBaseValues()

    def writeTo(self, fname):
        assert self.isFinalized
        fd = open(fname, 'w', encoding='utf-8')

        itemCount = self.lastBaseValue+1 # 0-based indexing
        fd.write('%d\n' % itemCount)

        for (i, term) in self.indexToValueMap.items():
            if i > self.lastBaseValue: # don't write special values
                break
            fd.write('%s %d\n' % (term, self.freq[term]))

        fd.close()
    
    '''
    Finalize indices and sort by descending frequency of each term, and then
    alphabetically
    Index 0 will be the most frequent term
    '''
    def finalizeBaseValues(self):
        assert not self.isFinalized
        round1Items = []

        allFreqs = set()
        for (termName, termFreq) in self.freq.items():
            round1Items.append((termFreq, termName))
            allFreqs.add(termFreq)
        allFreqs = list(allFreqs)
        allFreqs.sort(reverse=True)

        sortFinal = []
        # iterate frequencies in descending order
        for f in allFreqs:
            round2Tmp = []
            # find all items with this frequency and sort them by name,
            # ascending
            for (termFreq, termName) in round1Items:
                if termFreq == f:
                    round2Tmp.append(termName)
            assert len(round2Tmp) > 0, 'term not found'
            round2Tmp.sort()
            # append all term names with this frequency, sorted by name
            sortFinal += round2Tmp

        assert len(self.freq) == len(sortFinal), 'missing items detected'
        assert len(set(sortFinal)) == len(sortFinal), 'duplicates detected'

        i = 0
        self.indexToValueMap = dict()
        for v in sortFinal:
            self.indexToValueMap[i] = v
            self.lastBaseValue = i
            i += 1

        self.valueToIndexMap = dict()
        for (i, v) in self.indexToValueMap.items():
            self.valueToIndexMap[v] = i

        assert len(self.valueToIndexMap) == len(self.indexToValueMap), \
            'index<->value map length mismatch'
        self.isFinalized = True
