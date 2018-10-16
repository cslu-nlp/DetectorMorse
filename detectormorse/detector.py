# Copyright (c) 2014 Kyle Gorman <gormanky@ohsu.edu>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
from collections import namedtuple
from re import finditer, match, search

import pkg_resources
from nlup import case_feature, isnumberlike, listify, \
    BinaryAveragedPerceptron, BinaryConfusion, JSONable

from .ptbtokenizer import word_tokenize

# FIXME(kbg) can surely avoid full-blown tokenization


# defaults

NOCASE = False  # disable case-based features?
EPOCHS = 20     # number of epochs (iterations for classifier training)
BUFSIZE = 32    # for reading in left and right contexts...see below
CLIP = 8        # clip numerical count feature values
DEFAULT_MODEL = 'DM-wsj.json.gz'

# character classes

VOWELS = frozenset("AEIOUY")

# token classes

QUOTE_TOKEN = "*QUOTE*"
NUMBER_TOKEN = "*NUMBER*"

# regexes

PUNCT = r"((\.+)|([!?]))"
TARGET = PUNCT + r"(['`\")}\]]*)(\s+)"

LTOKEN = r"(\S+)\s*$"
RTOKEN = r"^\s*(\S+)"
NEWLINE = r"^\s*[\r\n]+\s*$"

QUOTE = r"^['`\"]+$"

# other

Observation = namedtuple("Observation", ["L", "P", "R", "B", "end"])


def slurp(filename, encoding='utf-8'):
    """
    Given a `filename` string, slurp the whole file into a string
    """
    with open(filename, encoding=encoding) as source:
        return source.read()


def load_from_resource(name):
    """
    Return a Detector loaded from resource with the specified name.

    The model name must match a filename existing under /models
    in this package.
    """
    # Note that you do not want os.path.join here as all resource paths
    # use forward slash
    filename = pkg_resources.resource_filename(__name__, 'models/' + name)
    return Detector.load(filename)


def default_model():
    """
    Return a Detector loaded from the default model.

    Currently, the default model is trained on WSJ.
    """
    return load_from_resource(DEFAULT_MODEL)


class Detector(JSONable):

    def __init__(self, text=None, nocase=NOCASE, epochs=EPOCHS,
                 classifier=BinaryAveragedPerceptron, **kwargs):
        self.classifier = classifier(**kwargs)
        self.nocase = nocase
        if text:
            self.fit(text, epochs)

    def __repr__(self):
        return "{}(classifier={!r})".format(self.__class__.__name__,
                                            self.classifier)

    # identify candidate regions

    @staticmethod
    def candidates(text):
        """
        Given a `text` string, get candidates and context for feature
        extraction and classification
        """
        for Pmatch in finditer(TARGET, text):
            # the punctuation mark itself
            P = Pmatch.group(1)
            # is it a boundary?
            B = bool(match(NEWLINE, Pmatch.group(5)))
            # L & R
            start = Pmatch.start()
            end = Pmatch.end()
            Lmatch = search(LTOKEN, text[max(0, start - BUFSIZE):start])
            if not Lmatch:  # this happens when a line begins with '.'
                continue
            L = word_tokenize(" " + Lmatch.group(1))[-1]
            Rmatch = search(RTOKEN, text[end:end + BUFSIZE])
            if not Rmatch:  # this happens at the end of the file, usually
                continue
            R = word_tokenize(Rmatch.group(1) + " ")[0]
            # complete observation
            yield Observation(L, P, R, B, end)

    # extract features

    @listify
    def extract_one(self, L, P, R):
        """
        Given left context `L`, punctuation mark `P`, and right context
        R`, extract features. Probability distributions for any
        quantile-based features will not be modified.
        """
        yield "*bias*"
        # L feature(s)
        if match(QUOTE, L):
            L = QUOTE_TOKEN
        elif isnumberlike(L):
            L = NUMBER_TOKEN
        else:
            yield "len(L)={}".format(min(len(L), CLIP))
            if "." in L:
                yield "L:*period*"
            if not self.nocase:
                cf = case_feature(R)
                if cf:
                    yield "L:{}'".format(cf)
            L = L.upper()
            if not any(char in VOWELS for char in L):
                yield "L:*no-vowel*"
        L_feat = "L='{}'".format(L)
        yield L_feat
        # P feature(s)
        yield "P='{}'".format(P)
        # R feature(s)
        if match(QUOTE, R):
            R = QUOTE_TOKEN
        elif isnumberlike(R):
            R = NUMBER_TOKEN
        else:
            if not self.nocase:
                cf = case_feature(R)
                if cf:
                    yield "R:{}'".format(cf)
            R = R.upper()
        R_feat = "R='{}'".format(R)
        yield R_feat
        # the combined L,R feature
        yield "{},{}".format(L_feat, R_feat)

    # actual detector operations

    def fit(self, text, epochs=EPOCHS):
        """
        Given a string `text`, use it to train the segmentation classifier
        for `epochs` iterations.
        """
        logging.debug("Extracting features and classifications.")
        Phi = []
        Y = []
        for (L, P, R, gold, _) in Detector.candidates(text):
            Phi.append(self.extract_one(L, P, R))
            Y.append(gold)
        self.classifier.fit(Y, Phi, epochs)
        logging.debug("Fitting complete.")

    def predict(self, L, P, R):
        """
        Given an left context `L`, punctuation mark `P`, and right context
        `R`, return True iff this observation is hypothesized to be a
        sentence boundary.
        """
        phi = self.extract_one(L, P, R)
        return self.classifier.predict(phi)

    def segments(self, text, strip=True):
        """
        Given a string of `text`, return a generator yielding each
        hypothesized sentence string
        """
        start = 0
        for (L, P, R, B, end) in Detector.candidates(text):
            if self.predict(L, P, R):
                sent = text[start:end]
                if strip:
                    sent = sent.rstrip()
                yield sent
                start = end
            # otherwise, there's probably not a sentence boundary here
        sent = text[start:]
        if strip:
            sent = sent.rstrip()
        yield sent

    def evaluate(self, text):
        """
        Given a string of `text`, compute confusion matrix for the
        classification task.
        """
        cx = BinaryConfusion()
        for (L, P, R, gold, _) in Detector.candidates(text):
            guess = self.predict(L, P, R)
            cx.update(gold, guess)
            if not gold and guess:
                logging.debug("False pos.: L='{}', R='{}'.".format(L, R))
            elif gold and not guess:
                logging.debug("False neg.: L='{}', R='{}'.".format(L, R))
        return cx
