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
Penn Treebank tokenizer, adapted from `nltk.tokenize.treebank.py`, which
in turn is adapted from an infamous sed script by Robert McIntyre. Even
ignoring the reduced import overhead, this is about half again faster than
the NLTK version; don't ask me why.

>>> s = '''Good muffins cost $3.88\\nin New York.  Please buy me\\ntwo of them.\\nThanks.'''
>>> word_tokenize(s)
['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.']
>>> s = "They'll save and invest more."
>>> word_tokenize(s)
['They', "'ll", 'save', 'and', 'invest', 'more', '.']
"""

from re import sub


RULES1 = [  # starting quotes
    (r'^\"', r'``'),
    (r'(``)', r' \1 '),
    (r'([ (\[{<])"', r'\1 `` '),
    # punctuation
    (r'([:,])([^\d])', r' \1 \2'),
    (r'\.\.\.', r' ... '),
    (r'[;@#$%&]', r' \g<0> '),
    (r'([^\.])(\.)([\]\)}>"\']*)\s*$', r'\1 \2\3 '),
    (r'[?!]', r' \g<0> '),
    (r"([^'])' ", r"\1 ' "),
    # parens, brackets, etc.
    (r'[\]\[\(\)\{\}\<\>]', r' \g<0> '),
    (r'--', r' -- ')]

# ending quotes
RULES2 = [(r'"', " '' "),
          (r'(\S)(\'\')', r'\1 \2 ')]

# all replaced with r"\1 \2 "
CONTRACTIONS = [r"(?i)([^' ])('S|'M|'D|') ",
                r"(?i)([^' ])('LL|'RE|'VE|N'T) ",
                r"(?i)\b(CAN)(NOT)\b",
                r"(?i)\b(D)('YE)\b",
                r"(?i)\b(GIM)(ME)\b",
                r"(?i)\b(GON)(NA)\b",
                r"(?i)\b(GOT)(TA)\b",
                r"(?i)\b(LEM)(ME)\b",
                r"(?i)\b(MOR)('N)\b",
                r"(?i)\b(WAN)(NA) ",
                r"(?i) ('T)(IS)\b",
                r"(?i) ('T)(WAS)\b"]


def word_tokenize(text):
    """
    Split string `text` into word tokens using the Penn Treebank rules
    """
    for (regexp, replacement) in RULES1:
        text = sub(regexp, replacement, text)
    # add extra space to make things easier
    text = " " + text + " "
    for (regexp, replacement) in RULES2:
        text = sub(regexp, replacement, text)
    for regexp in CONTRACTIONS:
        text = sub(regexp, r"\1 \2 ", text)
    # split and return
    return text.split()
