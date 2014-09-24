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

from collections import defaultdict, namedtuple
from re import escape, finditer, match, search, sub
from string import ascii_lowercase, ascii_uppercase, digits

from nltk import word_tokenize

from jsonable import JSONable
from decorators import listify, IO
from confusion import BinaryConfusion
from quantile import quantile_breaks, get_quantile
from perceptron import BinaryAveragedPerceptron as CLASSIFIER


LOGGING_FMT = "%(module)s: %(message)s"

# defaults
NOCASE = False  # disable case-based features?
ADD_N = 1       # Laplace smoothing constant
BINS = 5        # number of quantile bins (for discretizing features)
EPOCHS = 20     # number of epochs (iterations for classifier training)

BUFFER_SIZE = 128  # for reading in left and right contexts...see below

# character classes

DIGITS = frozenset(digits)
LOWERCASE = frozenset(ascii_lowercase)
UPPERCASE = frozenset(ascii_uppercase)
LETTERS = LOWERCASE | UPPERCASE
VOWELS = frozenset("AEIOU")

# regular expressions

PUNCT = frozenset(".!?")
TARGET = r"([{}])([\'\`\"]*)(\s+)".format(escape("".join(PUNCT)))

LTOKEN = r"\S+$"
RTOKEN = r"^\S+"
NEWLINE = r"^\s*[\r\n]+\s*$"

NUMBER = r"^(\-?\$?)(\d+(\,\d{3})*([\-\/\:\.]\d+)?)\%?$"
QUOTE = r"[\"\'\`]+"

# special tokens

QUOTE_TOKEN = "*QUOTE*"
NUMBER_TOKEN = "*NUMBER*"

observation = namedtuple("observation", ["L", "P", "Q", "S", "R", "end"])


@IO
def slurp(filename):
    """
    Given a `filename` string, slurp the whole file into a string
    """
    with open(filename, "r") as source:
        return source.read()


class Detector(JSONable):

    def __init__(self, text=None, nocase=NOCASE, epochs=EPOCHS,
                 bins=BINS, classifier=CLASSIFIER, **kwargs):
        self.classifier = classifier(**kwargs)
        if text:
            self.fit(text, nocase, epochs, bins)

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
            (P, Q, S) = Pmatch.groups()
            start = Pmatch.start()
            end = Pmatch.end()
            Lmatch = search(LTOKEN,
                            text[max(0, start - BUFFER_SIZE):start])
            if not Lmatch:  # this happens when a line begins with '.'
                continue
            L = word_tokenize(" " + Lmatch.group())[-1]
            Rmatch = search(RTOKEN, text[end:end + BUFFER_SIZE])
            if not Rmatch:  # this happens at the end of the document
                continue
            R = word_tokenize(Rmatch.group() + " ")[0]
            yield observation(L, P, Q, S, R, end)

    # extract features

    @staticmethod
    def token_case(string):
        """
        Compute one of six "cases" for a token:
        * number: is the NUMBER_TOKEN or contains a digit
        * punctuation: is the QUOTE_TOKEN or contains no alphabetic
          characters
        * upper: all alphabetic characters are uppercase
        * lower: all alphabetic characters are lowercase
        * title: first alphabetic character uppercase, later alphabetic
                 characters lowercase
        * mixed: none of the above
        """
        # TODO possible improvements/tweaks:
        #     * should we just look at the first letter, merging mixed with
        #       upper and lower, and merging upper and title?
        #     * should there be 2 features, one for the first letter and
        #       another for the rest-of-word?
        #     * should this be an instance method
        if string == NUMBER_TOKEN or any(ch in DIGITS for ch in string):
            return "number"
        # remove non-alphabetic characters
        rstring = "".join(ch for ch in string if ch in LETTERS)
        # if there aren't any, it's punctuation
        if string == QUOTE_TOKEN or not rstring:
            return "punctuation"
        if rstring[0] in UPPERCASE:
            if all(char in UPPERCASE for char in rstring[1:]):
                return "upper"
                # note that one-letter tokens will pass this test, and so
                # be coded as uppercase, not titlecase
            elif all(char in LOWERCASE for char in rstring[1:]):
                return "title"
            # mixed
        if all(char in LOWERCASE for char in rstring):
            return "lower"
        return "mixed"

    @listify
    def extract_one(self, L, P, Q, R, nocase=NOCASE):
        """
        Given an observation (left context `L`, punctuation marker `P`,
        quote string `Q`, and right context `R`), extract the classifier
        features. Probability distributions for decile-based features
        will not be modified.
        """
        yield "(bias)"
        # is L followed by one or more quotes?
        if Q:
            yield "(quote)"
        # L features
        if match(QUOTE, L):
            L = QUOTE_TOKEN
        elif match(NUMBER, L):
            L = NUMBER_TOKEN
        else:
            if not nocase:
                yield "case(L)={}".format(Detector.token_case(L))
            L = L.upper()
            yield "len(L)={}".format(min(10, len(L)))
            if not any(char in VOWELS for char in L):
                yield "(L:no-vowel)"
            if "." in L:
                yield "(L:period)"
        if L in self.q_Lfinal:
            yield "quantile(p(final|L))={}".format(self.q_Lfinal[L])
        # the P identity feature
        yield "P='{}'".format(P)
        L_feat = "L='{}'".format(L)
        yield L_feat
        # R features
        if match(QUOTE, R):
            R = QUOTE_TOKEN
        elif match(NUMBER, R):
            R = NUMBER_TOKEN
        else:
            if not nocase:
                R_case_feat = Detector.token_case(R)
                yield "case(R)={}".format(R_case_feat)
                # FIXME feature idea
                #yield "L='{}',case(R)={}".format(L_feat, R_case_feat)
            R = R.upper()
            yield "len(R)={}".format(len(R))
        if R in self.q_Rinitial:
            yield "quantile(p(initial|R))={}".format(self.q_Rinitial[R])
        if R in self.q_R0upper:
            yield "quantile(p(uppercase|R))={}".format(self.q_R0upper[R])
        R_feat = "R='{}'".format(R)
        yield R_feat
        yield "{},{}".format(L_feat, R_feat)

    # helpers for `fit`

    @staticmethod
    def _fit_merge_token(token):
        """
        Merge tokens as per `fit`
        """
        if match(QUOTE, token):
            return QUOTE_TOKEN
        if match(NUMBER, token):
            return NUMBER_TOKEN
        return token

    @staticmethod
    def _fit_freqs2quants(cfd, add_n=ADD_N, bins=BINS):
        """
        Convert a dictionary representing a conditional binomial
        probability distribution `cfd` to a dictionary containing the
        indices of the nearest quantile. To enable Laplace smoothing,
        set `add_n` to some positive value (perhaps .5 or 1). The
        parameter `bins` determines the number of quantiles.
        """
        retval = {}
        # convert to probability distribution
        for (token, cfd_token) in cfd.items():
            numerator = cfd_token[True] + add_n
            denominator = numerator + cfd_token[False] + add_n
            retval[token] = numerator / denominator
        # convert probabilities with quantile values
        Qb = quantile_breaks(retval.values(), bins)
        retval = {token: get_quantile(Qb, value) for
                   (token, value) in retval.items()}
        return retval

    @staticmethod
    def _fit_first_middle_last(line):
        tokens = [Detector._fit_merge_token(token) for
                                            token in word_tokenize(line)]
        if not tokens:
            return (None, [], None)
        first = tokens.pop(0)
        if not tokens:
            return (first, [], None)
        last = tokens.pop()
        if last in PUNCT:
            if not tokens:
                return (first, [], None)
            last = tokens.pop()
        return (first, tokens, last)

    @staticmethod
    def _fit_case(token):
        if not token:
            return
        if token[0] in UPPERCASE:
            return True
        if token[0] in LOWERCASE:
            return False

    # actual detector operations

    def fit(self, text, nocase=NOCASE, epochs=EPOCHS, bins=BINS):
        """
        Given a string `text`, use it to train the segmentation classifier
        for `epochs` iterations.
        """
        logging.debug("Computing quantiles for probabilistic features.")
        # compute conditional frequencies
        binomial = lambda: {True: 0, False: 0}
        f_Lfinal = defaultdict(binomial)
        f_Rinitial = defaultdict(binomial)
        f_R0upper = defaultdict(binomial)
        for line in text.splitlines():
            (first, middle, last) = Detector._fit_first_middle_last(line)
            if not first:
                continue
            first_f = first.upper()
            f_Lfinal[first_f][False] += 1
            f_Rinitial[first_f][True] += 1
            # note that we don't do R0upper initially, as case in this 
            # position is inherently ambiguous
            # middle
            for token in middle:
                token_f = token.upper()
                case = Detector._fit_case(token)
                if case is not None:
                    f_R0upper[token_f][case] += 1
                f_Lfinal[token_f][False] += 1
                f_Rinitial[token_f][False] += 1
            # last
            if not last:
                continue
            last_f = last.upper()
            case = Detector._fit_case(last)
            if case is not None:
                f_R0upper[last_f][case] += 1
            f_Lfinal[last_f][True] += 1
            f_Rinitial[last_f][False] += 1
        self.q_Lfinal = Detector._fit_freqs2quants(f_Lfinal, bins=bins)
        self.q_Rinitial = Detector._fit_freqs2quants(f_Rinitial, bins=bins)
        self.q_R0upper = Detector._fit_freqs2quants(f_R0upper, bins=bins)
        logging.debug("Extracting features and classifications.")
        X = []
        Y = []
        for (L, P, Q, S, R, _) in Detector.candidates(text):
            X.append(self.extract_one(L, P, Q, R, nocase))
            Y.append(bool(match(NEWLINE, S)))
        self.classifier.fit(X, Y, epochs)
        logging.debug("Fitting complete.")

    def predict(self, L, P, Q, R, nocase=NOCASE):
        """
        Given an observation (left context `L`, punctuation marker `P`,
        quote string `Q`, and right context `R`), return True iff
        this observation is hypothesized to be a sentence boundary.

        This presumes the model has already been fit.
        """
        x = self.extract_one(L, P, Q, R, nocase)
        return self.classifier.predict(x)

    def segments(self, text, nocase=NOCASE):
        """
        Given a string of `text`, return a generator yielding each
        hypothesized sentence string
        """
        start = 0
        for (L, P, Q, S, R, end) in Detector.candidates(text):
            # if there's already a newline there, we have nothing to do
            if match(NEWLINE, S):
                continue
            if self.predict(L, P, Q, R, nocase):
                yield text[start:end]
                start = end
            # otherwise, there's probably not a sentence boundary here
        yield text[start:]

    def evaluate(self, text, nocase=NOCASE):
        """
        Given a string of `text`, compute confusion matrix for the
        classification task.
        """
        cx = BinaryConfusion()
        for (L, P, Q, S, R, _) in Detector.candidates(text):
            gold = bool(match(NEWLINE, S))
            guess = self.predict(L, P, Q, R, nocase)
            cx.update(gold, guess)
        return cx


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
    argparser.add_argument("-B", "--bins", type=int, default=BINS,
                           help="# of bins (default: {})".format(BINS))
    argparser.add_argument("-E", "--epochs", type=int, default=EPOCHS,
                           help="# of epochs (default: {})".format(EPOCHS))
    argparser.add_argument("-C", "--nocase", action="store_true",
                           help="disable case features")
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
        detector = Detector(slurp(args.train), bins=args.bins,
                            epochs=args.epochs, nocase=args.nocase)
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
        if args.verbose or args.really_verbose:
            cx.pprint()
        print(cx.summary)
