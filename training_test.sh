#!/bin/sh
# the sample provided training corpus contains 931 feature bags with this configuration
mkdir -p /tmp/testmodelEN
cp parser-config.sample /tmp/testmodelEN/parser-config
python3 dep_parser.py /tmp/testmodelEN corpus/en-ud-train.conllu  corpus/en-ud-test.conllu  --train --epochs 5 --restart
