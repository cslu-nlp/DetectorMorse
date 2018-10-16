Detector Morse
==============

Detector Morse is a program for sentence boundary detection (henceforth, SBD), also known as sentence segmentation. Consider the following sentence, from the Wall St. Journal portion of the Penn Treebank:

    Rolls-Royce Motor Cars Inc. said it expects its U.S. sales to remain
    steady at about 1,200 cars in 1990.

This sentence contains 4 periods, but only the last denotes a sentence boundary. The first one in `U.S.` is unambiguously part of an acronym, not a sentence boundary; the same is true of expressions like `$12.53`. But the periods at the end of `Inc.` and `U.S.` could easily denote a sentence boundary. Humans use the local context to determine that neither period denote sentence boundaries (e.g. the selectional properties of the verb _expect_ are not met if there is a sentence bounary immediately after `U.S.`). Detector Morse uses artisinal, handcrafted contextual features and low-impact, leave-no-trace machine learning methods to automatically detect sentence boundaries.

SBD is one of the earliest pieces of many natural language processing pipelines. Since errors at this step are likely to propagate, SBD is an important---albeit overlooked---problem in natural language processing.

Detector Morse has been tested on CPython 3.4 and PyPy3 (2.3.1, corresponding to Python 3.2); the latter is much faster. Detector Morse depends on the Python module `nlup` (which in turn relies on `jsonpickle`) to (de)serialize models. For the versions used, see `requirements.txt`.

Installation
============

```
pip install detectormorse
```

Usage
=====

```
Detector Morse, by Kyle Gorman
     
usage: python -m detectormorse [-h] [-v | -V] (-t TRAIN | -r [READ])
                               (-s SEGMENT | -w WRITE | -e EVALUATE)
                               [-E EPOCHS] [-C] [--preserve-whitespace]

Detector Morse

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         enable verbose output
  -V, --really-verbose  enable even more verbose output
  -t TRAIN, --train TRAIN
                        training data
  -r [READ], --read [READ]
                        read in a serialized model from a path or read the
                        default model if no path is specified
  -s SEGMENT, --segment SEGMENT
                        segment sentences
  -w WRITE, --write WRITE
                        write out serialized model
  -e EVALUATE, --evaluate EVALUATE
                        evaluate on segmented data
  -E EPOCHS, --epochs EPOCHS
                        # of epochs (default: 20)
  -C, --nocase          disable case features
  --preserve-whitespace
                        preserve whitespace when segmenting
```

Files used for training (`-t`/`--train`) and evaluation (`-e`/`--evaluate`) should contain one sentence per line; newline characters are ignored otherwise.

When segmenting a file (`-s`/`--segment`), DetectorMorse simply inserts a newline after predicted sentence boundaries that aren't already marked by one. All other newline characters are passed through, unmolested.

The included `DM-wsj.json.gz` is a segmenter model trained on the Wall St. Journal portion of the Penn Treebank. This model can be loaded by using `detector.default_model()` or by specifying `-r` with no path at the command line.

Method
======

See [this blog post](http://www.wellformedness.com/blog/simpler-sentence-boundary-detection/).

Caveats
=======

DetectorMorse processes text by reading the entire file into memory. This means it will not work with files that won't fit into the available RAM. The easiest way to get around this is to import the `Detector` instance in your own Python script.

Exciting extras!
================

I've included a Perl script `untokenize.pl` which attempts to invert the Penn Treebank tokenization process. Tokenization is an inherently "lossy" procedure, so there is no guarantee that the output is exactly how it appeared in the WSJ. But, the rules appear to be correct and produce sane text, and I have used it for all experiments. **Update (2015-02-10): I've removed this script; I just use the Stanford tokenizer for this purpose, now.**
