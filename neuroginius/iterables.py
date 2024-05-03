from functools import reduce
import sys
from itertools import filterfalse, groupby, combinations, chain, product
from operator import add

def unique(iterable):
    """
    List unique elements, preserving order. Remember all elements ever seen.
    Siplified version of unique_everseen from itertools recipes.
    """
    seen = set()
    for element in filterfalse(seen.__contains__, iterable):
        seen.add(element)
        yield element

def itersize(iterable):
    return reduce(
        add,
        map(sys.getsizeof, iterable)
    )

def all_equal(iterable):
    """
    From itertools recipes
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def loops(it):
    unique_el = unique(it)
    return ((el, el) for el in unique_el)

def all_connectivities(it):
    return chain(combinations(it, 2), loops(it))

def join(it):
    return (f"{i}_{j}" for i, j in it)