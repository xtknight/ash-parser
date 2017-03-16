#!/bin/sh
rm -f /tmp/testmodelEN/model.ckpt.* /tmp/testmodelEN/checkpoint
mkdir -p /tmp/testmodelEN
cp parser-config.sample /tmp/testmodelEN/
python3 dep_parser.py /tmp/testmodelEN corpus/en-ud-train.conllu  corpus/en-ud-test.conllu  --train
