# ash-parser

This was originally for a class project.

Utilizes a [Chen and Manning (2014)](http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf) style neural network parser in Python and TensorFlow. Many elements mimic [SyntaxNet](https://github.com/tensorflow/models/tree/master/syntaxnet).

I analyze [SyntaxNet's Architecture](http://andrewmatteson.name/index.php/2017/02/04/inside-syntaxnet/) here.

parsing-config file is required to be created in the model directory before execution.

Run training_test.sh for an example of how to train a model. Evaluation during training works as well, but there is no API for tagging new input yet or serving a model.

External dependencies
- NumPy
- TensorFlow 1.0

Similarities to SyntaxNet
- Same embedding system (configurable per-feature group deep embedding)
- Same optimizer (Momentum with exponential moving average)
- Lexicon builder is identical for words, tags, and labels
- Map files output by SyntaxNet and AshParser should be identical
- Evaluation metric is identical (SyntaxNet's corresponds to AshParser's UAS)
- Feature system is almost identical (except perhaps some very rare corner cases)
- Due to same architecture, accuracy should be very close to Greedy SyntaxNet

Differences from SyntaxNet:
- Supports GPU: training phase can complete in minutes
- Pure Python3 implementation. No need for bazel
- LAS (Labeled Attachment Score) prints out during evaluation
- Precalculation and caching of feature bags. This makes it easier to train multiple models with the same token features but different hyperparameters
- No support for structured (beam) parsing. Considering LSTM or something simpler and faster instead
- Feature groups are automatically created by groups of tag, word, and label rather than by grouping together with semicolon in a context file
- Only support for the arc-standard transition parser, not the POS tagger, morphological analyzer, or tokenizer
- ngrams, punctuation_amount, morph tags and other features not yet implemented
