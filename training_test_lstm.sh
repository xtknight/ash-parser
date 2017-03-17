#!/bin/sh
# the sample provided training corpus contains 931 feature bags with this configuration
python3 dep_parser_lstm.py /tmp/lstmtest corpus/en-ud-train.conllu  corpus/en-ud-test.conllu  --train --epochs 5 --restart
