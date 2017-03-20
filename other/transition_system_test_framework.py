from conll_utils import *
from parser_state import ParserState
from arc_standard_transition_system import ArcStandardTransitionState, \
     ArcStandardTransitionSystem
from arc_eager_transition_system import ArcEagerTransitionState, \
     ArcEagerTransitionSystem
from gold_parse_reader import GoldParseReader
from decoded_parse_reader import DecodedParseReader
from lexicon import Lexicon
from model_parameters import ModelParameters
import logging

logger = logging.getLogger('TransitionSystemTest')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# arc-standard, arc-eager
system = 'arc-eager'
trainingFile = '/home/andy/Downloads/arcs-py/sample-data/train.parses'

if system == 'arc-standard':
    transition_system = ArcStandardTransitionSystem()
    transition_system_class = ArcStandardTransitionSystem
    transition_state_class = ArcStandardTransitionState
elif system == 'arc-eager':
    transition_system = ArcEagerTransitionSystem()
    transition_system_class = ArcEagerTransitionSystem
    transition_state_class = ArcEagerTransitionState
else:
    assert None, 'transition system must be arc-standard or arc-eager'


def dynamicOracleTrainTest(parser_state):
    LUT = ['SHIFT', 'RIGHT', 'LEFT', 'REDUCE']

    n = 0
    while not transition_system.isFinalState(parser_state):
        n += 1
        legal_transitions = transition_system_class.legal(parser_state)
        print('LEGAL ', ' '.join([LUT[p] for p in legal_transitions]))
        zero_cost = transition_system_class.dynamicOracle(parser_state,
            legal_transitions)
        print(str(n) + ' [ ' + ' '.join([LUT[z] for z in zero_cost]) + ' ]')

        if len(zero_cost) == 0:
            raise Exception('no zero cost')

        ## TODO: make it actually perform operation
        transition_system.performShift(parser_state) # FOR TESTING

def __main__():
    trainingCorpus = ParsedConllFile(keepMalformed=False,
        projectivize=True)

    trainingCorpus.read( \
        open(trainingFile, 'r',
            encoding='utf-8').read())

    # make fake model params, enough for lexicon builder
    # we still need feature_maps to use ParserState
    modelParams = ModelParameters('')
    modelParams.trainingFile = trainingFile
    modelParams.cfg = {'projectivizeTrainingSet': True}

    lexicon = Lexicon(modelParams)
    lexicon.compute()

    sentence = trainingCorpus.sentences[0]

    parser_state = ParserState(sentence,
                               lexicon.getFeatureMaps())

    # necessary for initializing and pushing root
    # (only initialize transition_state_class once!)
    # keep arc_state in sync with parser_state
    arc_state = transition_state_class(parser_state)

    dynamicOracleTrainTest(parser_state)

__main__()
