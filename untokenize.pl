#!/usr/bin/perl
# untokenize.pl: undo Penn Treebank tokenization
# Kyle Gorman <gormanky@ohsu.edu>
# 
# no copyright claimed: this is just too simplistic


use strict;
use warnings;

my $replace = "";                   # replacement string, usually null
my $holder = "<rAnD0mPlaceHolder>"; # placeholder string, to be removed

while (<>) {
    s/ ([:;,\.\?!%]|'')/$holder$1/g;
    s/ ('m|'d|'s|n't|'ll|'ve|'re)/$holder$1/gi;
    s/(``) /$1$holder/g;
    s/(?<=\s)(\$) /$1$holder/g;  # don't in cases like "US$ "
    s/ (' )/$holder$1/g;
    s/ (--) /$holder$1$holder/g;
    s/(\(|-LRB-) /\($holder/g;
    s/(\[|-LSB-) /\[$holder/g;
    s/(\{|-LCB-) /\{$holder/g;
    s/ (\)|-RRB-)/\)$holder/g;
    s/ (\]|-RSB-)/\]$holder/g;
    s/ (\}|-RCB-)/\}$holder/g;
    s/\\\//\//g;
    s/(?<=\.$holder)\.//g;       # fix occasional "U.S.." examples
    s/$holder(\.\.\.)/ $1/g;
    s/$holder/$replace/g;
    print;
}
