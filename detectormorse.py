#!/usr/bin/env python
#
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

"""
DetectorMorse: supervised sentence boundary detection
"""


import logging

from nltk import word_tokenize
from collections import namedtuple
from re import finditer, match, search
from string import ascii_lowercase, ascii_uppercase, digits

from jsonable import JSONable
from decorators import listify, IO
from confusion import BinaryConfusion
from quantile import quantiles, nearest_quantile
from perceptron import BinaryAveragedPerceptron as BinaryClassifier


LOGGING_FMT = "%(module)s: %(message)s"

# defaults
NOCASE = False
T = 20

# for reading in left and right contexts...see below
BUFFER_SIZE = 128

# globals
DIGITS = frozenset(digits)
LOWERCASE = frozenset(ascii_lowercase)
UPPERCASE = frozenset(ascii_uppercase)
LETTERS = LOWERCASE | UPPERCASE
VOWELS = frozenset("AEIOU")
SURETHANG = frozenset("!?")

# regular expressions
TARGET = r"([\.\!\?])([\'\`\"]*)(\s+)"
LTOKEN = r"\S+$"
RTOKEN = r"^\S+"
NEWLINE = r"^\s*[\r\n]+\s*$"
NUMBER = r"^(\-?\$?)(\d+(\,\d{3})*([\-\/\:\.]\d+)?)\%?$"
QUOTE = r"[\"\'\`]+"

# special tokens
NUMBER_TOKEN = "*NUMBER*"
QUOTE_TOKEN = "*QUOTE*"

observation = namedtuple("observation", ["L", "P", "Q", "S", "R", "end"])


# helpers

@IO
def slurp(filename):
    """
    Given a `filename` string, slurp the whole file into a string
    """
    with open(filename, "r") as source:
        return source.read()

def candidates(text):
    """
    Given a `text` string, get candidates and context for feature 
    extraction and classification
    """
    for Pmatch in finditer(TARGET, text):
        (P, Q, S) = Pmatch.groups()
        start = Pmatch.start()
        end = Pmatch.end()
        Lmatch = search(LTOKEN, text[max(0, start - BUFFER_SIZE):start])
        if not Lmatch:
            # this usually happens when a line begins with '.'
            continue
        L = word_tokenize(" " + Lmatch.group())[-1]
        Rmatch = search(RTOKEN, text[end:end + BUFFER_SIZE])
        if not Rmatch: 
            # this usually happens at the end of the document
            continue
        R = word_tokenize(Rmatch.group() + " ")[0]
        yield observation(L, P, Q, S, R, end)


def token_case(string):
    """
    Compute one of six "cases" for a token:

    * upper: all alphabetic characters are uppercase
    * lower: all alphabetic characters are lowercase
    * title: first alphabetic character uppercase, later alphabetic
             characters lowercase
    * number: matches the reserved token *NUMBER*
    * punctuation: the token contains no alphabetic characters
    * mixed: none of the above
    """
    # TODO possible improvements/tweaks:
    #     * should we just look at the first letter, merging mixed with
    #       upper and lower, and merging upper and title?
    #     * should there be 2 features, the first-letter and rest-of-word?
    #     * should this be an instance method
    if string == "*NUMBER*" or any(char in DIGITS for char in string):
        return "number"
    # remove non-alphabetic characters
    rstring = "".join(char for char in string if char in LETTERS)
    # if there aren't any, it's punctuation
    if not rstring:
        return "punctuation"
    if rstring[0] in UPPERCASE:
        if all(char in UPPERCASE for char in rstring[1:]):
            return "upper"
            # note that one-letter tokens will pass this test, and so
            # be coded as uppercase, not titlecase
        elif all(char in LOWERCASE for char in rstring[1:]):
            return "title"
        # mixed
    elif all(char in LOWERCASE for char in rstring):
        return "lower"
    return "mixed"


class Detector(JSONable):

    def __init__(self, text=None, nocase=NOCASE, T=T,
                 classifier=BinaryClassifier):
        self.classifier = classifier()
        if text:
            self.fit(text, nocase, T)

    def __repr__(self):
        return "{}(classifier={!r})".format(self.__class__.__name__,
                                            self.classifier)

    @listify
    def extract_one(self, L, Q, R, nocase=NOCASE):
        """
        Extract features for a single observation of the form /L\.Q?\s+R/;
        Probability distributions for decile-based features will not be
        recomputed.
        """
        # FIXME decile-based features not yet implemented
        yield "(bias)"
        # is L followed by one or more quotes?
        if Q:
            yield "(quote)"
        # L features
        if match(NUMBER, L):
            L = NUMBER_TOKEN
        else:
            if not nocase:
                yield "case(L)={}".format(token_case(L))
            L = L.upper()
            yield "len(L)={}".format(len(L))
            if not any(char in VOWELS for char in L):
                yield "(L:no-vowel)"
            if "." in L:
                yield "(L:period)"
        Lfeat = "L='{}'".format(L)
        yield Lfeat
        # R features
        if match(NUMBER, R):
            R = NUMBER_TOKEN
        elif match(QUOTE, R):
            R = QUOTE_TOKEN
        else:
            if not nocase:
                yield "case(R)={}".format(token_case(R))
            R = R.upper()
            yield "len(R)={}".format(len(R))
        Rfeat = "R='{}'".format(R)
        yield Rfeat
        yield "{},{}".format(Lfeat, Rfeat)

    def fit(self, text, nocase=NOCASE, T=T):
        logging.debug("Extracting features from training data.")
        # FIXME compute decile-based features, too
        X = []
        Y = []
        for (L, P, Q, S, R, _) in candidates(text):
            if P in SURETHANG:
                continue
            X.append(self.extract_one(L, Q, R, nocase))
            Y.append(bool(match(NEWLINE, S)))
        self.classifier.fit(X, Y, T)

    def predict(self, L, Q, R, nocase=NOCASE):
        x = self.extract_one(L, Q, R, nocase)
        return self.classifier.predict(x)

    def segments(self, text, nocase=NOCASE):
        start = 0
        for (L, P, Q, S, R, end) in candidates(text):
            # if there's already a newline there, we have nothing to do
            if match(NEWLINE, S):
                continue
            if P in SURETHANG or self.predict(L, Q, R, nocase):
                yield text[start:end]
                start = end
            # otherwise, there's probably not a sentence boundary here
        yield text[start:]

    def evaluate(self, text, nocase=NOCASE):
        """
        Compute binary confusion matrix for classification task.
        NB: To enable comparability with other systems, we count
            "sure thang" (? or !) boundaries as true positives.
        """
        cx = BinaryConfusion()
        for (L, P, Q, S, R, _) in candidates(text):
            gold = bool(match(NEWLINE, S))
            guess = P in SURETHANG or self.predict(L, Q, R, nocase)
            cx.update(gold, guess)
        return cx

    """
    @staticmethod
    def freq2prob(freqs, smoothing=0):
        addin = smoothing * 2
        return {utoken: (freq[True] + smoothing) /
                        (freq[True] + freq[False] + addin) for \
                        (utoken, freq) in freqs.items()}

    def extract_features(self, lines):
        ## deciles for those two features that need them
        # compute LB and Rup frequencies
        LB_freqs = defaultdict(partial(defaultdict, int))
        Rup_freqs .Rup = defaultdict(partial(defaultdict, int))
        for line in lines:
            tokens = word_tokenize(text)
            L = len(tokens)
            for (i, token) in enumerate(tokens):
                utoken = token.upper()
                LB_freqs[utoken][i <= L] += 1
                Rup_freqs[utoken][token[0] in UPPER] += 1
        # convert to probabilities
        LB_probs = Detector.freqs2prob(LB_freqs)
        Rup_probs = Detector.freqs2prob(Rup_freqs)
        # convert to deciles
        self.LB_deciles = 
    """


if __name__ == "__main__":
    from argparse import ArgumentParser
    argparser = ArgumentParser(description="DetectorMorse, by Kyle Gorman")
    argparser.add_argument("-v", "--verbose", action="store_true",
                           help="enable verbose output")
    argparser.add_argument("-V", "--really-verbose", action="store_true",
                           help="enable even more verbose output")
    inp_group = argparser.add_mutually_exclusive_group(required=True)
    inp_group.add_argument("-t", "--train", help="training data")
    inp_group.add_argument("-r", "--read", help="read in serialized model")
    out_group = argparser.add_mutually_exclusive_group(required=True)
    out_group.add_argument("-s", "--segment", help="segment sentences")
    out_group.add_argument("-w", "--write",
                           help="write out serialized model")
    out_group.add_argument("-e", "--evaluate",
                           help="evaluate on segmented data")
    argparser.add_argument("-C", "--nocase", action="store_true",
                           help="disable case features")
    argparser.add_argument("-T", type=int, default=T,
                           help="# of epochs (default: {})".format(T))
    args = argparser.parse_args()
    # verbosity block
    if args.really_verbose:
        logging.basicConfig(format=LOGGING_FMT, level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(format=LOGGING_FMT, level=logging.INFO)
    else:
        logging.basicConfig(format=LOGGING_FMT)
    # input block
    detector = None
    if args.train:
        logging.info("Training model on '{}'.".format(args.train))
        detector = Detector(slurp(args.train), T=args.T,
                             nocase=args.nocase)
    elif args.read:
        logging.info("Reading pretrained model '{}'.".format(args.read))
        detector = IO(Detector.load)(args.read)
    # output block
    if args.segment:
        logging.info("Segmenting '{}'.".format(args.segment))
        for segment in detector.segments(slurp(args.segment), 
                                         nocase=args.nocase):
            print(segment)
    if args.write:
        logging.info("Writing model to '{}'.".format(args.write))
        IO(detector.dump)(args.write)
    elif args.evaluate:
        logging.info("Evaluating model on '{}'.".format(args.evaluate))
        cx = detector.evaluate(slurp(args.evaluate), nocase=args.nocase)
        cx.pprint()
        print(cx.summary)
