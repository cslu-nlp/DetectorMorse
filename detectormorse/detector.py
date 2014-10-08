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

from math import log # FIXME
from collections import defaultdict, namedtuple
from re import escape, finditer, match, search, sub
from string import ascii_lowercase, ascii_uppercase, digits

from nltk import word_tokenize

from nlup.confusion import BinaryConfusion
from nlup.decorators import listify, IO
from nlup.jsonable import JSONable
from nlup.perceptron import BinaryAveragedPerceptron as CLASSIFIER

from .quantile import quantile_breaks, get_quantile


# defaults
NOCASE = False  # disable case-based features?
ADD_N = 1       # Laplace smoothing constant
BINS = 10       # number of quantile bins (for discretizing features)
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

# inf

INF = float("inf")

# observation tuple

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
        self.nocase = nocase
        if text:
            self.fit(text, epochs, bins)

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

    @listify
    def extract_one(self, L, P, Q, R):
        """
        Given an observation (left context `L`, punctuation marker `P`,
        quote string `Q`, and right context `R`), extract the classifier
        features. Probability distributions for decile-based features
        will not be modified.
        """
        yield "(bias)"
        # is L followed by one or more quotes?
        """
        if Q:
            yield "(quote)"
        # L features
        if match(QUOTE, L):
            L = QUOTE_TOKEN
        elif match(NUMBER, L):
            L = NUMBER_TOKEN
        elif not any(char in LETTERS for char in L):
            pass # FIXME
            #yield "(L:punctuation)"
        else:
            if not self.nocase:
                if any(char in UPPERCASE for char in L):
                    yield "(L:uppercase)"
                    if L[0] in UPPERCASE:
                        yield "(L_0:uppercase)"
            L = L.upper()
            yield "len(L)={}".format(min(10, len(L)))
            if not any(char in VOWELS for char in L):
                yield "(L:no-vowel)"
            if "." in L:
                yield "(L:period)"
        if L in self.q_Lfinal:
            yield "quantile(p(final|L))={}".format(self.q_Lfinal[L])
        # FIXME
        #if L in self.c_Lfinal:
        #     yield "log(f(final|L))={}".format(self.c_Lfinal[L])
        # the P identity feature
        yield "P='{}'".format(P)
        L_feat = "L='{}'".format(L)
        yield L_feat
        # R features
        """
        if match(QUOTE, R):
            R = QUOTE_TOKEN
        elif match(NUMBER, R):
            R = NUMBER_TOKEN
        elif not any(char in LETTERS for char in L):
            pass # FIXME
            #yield "(R:punctuation)"
        else:
            """
            if not self.nocase:
                if any(char in UPPERCASE for char in R):
                    yield "(R:uppercase)"
                    if R[0] in UPPERCASE:
                        yield "(R_0:uppercase)"
            """
            R = R.upper()
            yield "len(R)={}".format(len(R))
        """
        if R in self.q_Rinitial:
            yield "quantile(p(initial|R))={}".format(self.q_Rinitial[R])
        if R in self.q_R0upper:
            yield "quantile(p(uppercase|R))={}".format(self.q_R0upper[R])
        # FIXME
        #if R in self.c_Rinitial:
        #    yield "log(f(initial|R))={}".format(self.c_Rinitial[R])
        #if R in self.c_R0upper:
        #    yield "log(f(uppercase|R))={}".format(self.c_R0upper[R])
        R_feat = "R='{}'".format(R)
        yield R_feat
        yield "{},{}".format(L_feat, R_feat)
        """

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

    # FIXME
    """
    @staticmethod
    def _fit_freqs2logcounts(cfd, base=2):
        retval = {}
        for (token, cfd_token) in cfd.items():
            count = cfd_token[True]
            retval[token] = min(10, int(log(count)) if count else -INF)
        return retval
    """

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

    def fit(self, text, epochs=EPOCHS, bins=BINS):
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
        # FIXME
        #self.c_Lfinal = Detector._fit_freqs2logcounts(f_Lfinal)
        #self.c_Rinitial = Detector._fit_freqs2logcounts(f_Rinitial)
        #self.c_R0upper = Detector._fit_freqs2logcounts(f_R0upper)
        logging.debug("Extracting features and classifications.")
        X = []
        Y = []
        for (L, P, Q, S, R, _) in Detector.candidates(text):
            X.append(self.extract_one(L, P, Q, R))
            Y.append(bool(match(NEWLINE, S)))
        self.classifier.fit(X, Y, epochs)
        logging.debug("Fitting complete.")

    def predict(self, L, P, Q, R):
        """
        Given an observation (left context `L`, punctuation marker `P`,
        quote string `Q`, and right context `R`), return True iff
        this observation is hypothesized to be a sentence boundary.

        This presumes the model has already been fit.
        """
        x = self.extract_one(L, P, Q, R)
        return self.classifier.predict(x)

    def segments(self, text):
        """
        Given a string of `text`, return a generator yielding each
        hypothesized sentence string
        """
        start = 0
        for (L, P, Q, S, R, end) in Detector.candidates(text):
            # if there's already a newline there, we have nothing to do
            if match(NEWLINE, S):
                continue
            if self.predict(L, P, Q, R):
                yield text[start:end]
                start = end
            # otherwise, there's probably not a sentence boundary here
        yield text[start:]

    def evaluate(self, text):
        """
        Given a string of `text`, compute confusion matrix for the
        classification task.
        """
        cx = BinaryConfusion()
        for (L, P, Q, S, R, _) in Detector.candidates(text):
            gold = bool(match(NEWLINE, S))
            guess = self.predict(L, P, Q, R)
            cx.update(gold, guess)
        return cx
