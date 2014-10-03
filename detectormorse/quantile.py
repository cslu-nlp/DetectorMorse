# Copyright (c) 2014 Kyle Gorman
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
quantile: quantile estimation methods

>>> x = [11.4, 17.3, 21.3, 25.9, 40.1, 50.5, 60., 70., 75.]
>>> qe = quantile_estimates(x, 5)
>>> qe[1]
18.1
>>> qb = quantile_breaks(x, 5)
>>> get_quantile(qb, 20)
1
"""

from math import modf
from bisect import bisect

from nlup.decorators import listify


@listify
def quantile_estimates(values, bins):
    """
    Compute sample quantile estimates for `bins` evenly spaced bins. The
    algorithm here is "#8" recommended in the following paper:

    R.J. Hyndman & Y. Fan. 1996. Sample quantiles in statistical
    packages. American Statistician 50(4): 361-365.

    The code here is loosely based off of the excellent code by:

    http://adorio-research.org/wordpress/?p=125
    """
    svalues = sorted(values)
    L = len(svalues)
    if L < 2 or bins < 2:
        return
    yield svalues[0]
    for i in range(1, bins):
        (g, j) = modf(1 / 3 + (L + 1 / 3) * (i / bins) - 1)
        j = int(j)
        if g == 0:
            yield svalues[j]
        else:
            yield svalues[j] + (svalues[j + 1] - svalues[j]) * g


@listify
def quantile_breaks(values, bins):
    """
    Compute breakpoints between quantiles, to be used as an argument to
    `nearest_quantile`.
    """
    estimates = quantile_estimates(values, bins)
    for (q0, q1) in zip(estimates, estimates[1:]):
        yield (q0 + q1) / 2
    

def get_quantile(quantiles, value):
    """
    Given a sorted list of estimated `quantiles`, compute the 0-based
    index of the closest quantile to an observed `value`
    """
    return bisect(quantiles, value)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
