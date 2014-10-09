#!/usr/bin/perl
# untokenize.pl: undo Penn Treebank tokenization
# Kyle Gorman <gormanky@ohsu.edu>

use strict;
use warnings;

my $replace = "";                   # replacement string, usually null
my $holder = "<rAnD0mPlaceHolder>"; # placeholder string, to be removed

while (<>) {
    # easy punctuation
    s/ ([:;,\.\?!%]|'')/$holder$1/g;
    # clitics
    s/ ('m|'d|'s|n't|'ll|'ve|'re)/$holder$1/gi;
    # left quotes
    s/(`+) /$1$holder/g;
    # difficult punctuation
    s/(?<=\s)(\$) /$1$holder/g;
    s/ ('+($holder| ))/$holder$1/g;
    s/ (--) /$holder$1$holder/g;
    # brackets
    s/(\(|-LRB-) /\($holder/g;
    s/(\[|-LSB-) /\[$holder/g;
    s/(\{|-LCB-) /\{$holder/g;
    s/ (\)|-RRB-)/\)$holder/g;
    s/ (\]|-RSB-)/\]$holder/g;
    s/ (\}|-RCB-)/\}$holder/g;
    # fraction
    s/\\\//\//g;
    # fix final \.\. (occasional "U.S.."
    s/(?<=\.$holder)\.//g;
    s/$holder(\.\.\.)/ $1/g;
    # replace placeholder string with replacement string
    s/$holder/$replace/g;
    print;
}
