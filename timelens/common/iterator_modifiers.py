def make_skip_iterator(iterator, number_of_skips):
    """Returns iterator that skips elements form the original iterator.
    
    E.g. if original iterator return 1, 2, 3, 4, the modified iterator 
    with "number_of_skips"=2 will return 1, 3. 
    """
    for item in iterator:
        yield item
        for _ in range(number_of_skips):
            next(iterator, None)

def make_iterator_over_groups(iterator, group_size):
    """Returns iterator, that returns groups of values from the original iterator.
    
    For example, if original iterator returns 1, 2, 3, 4, 5, "group_size"=3 and
    output iterator returns (1,2,3), (2,3,4).
    """
    group = []
    for item in iterator:
        group.append(item)
        if len(group) == group_size:
            yield tuple(group)
            group = group[1:] 

def make_iterator_with_repeats(iterator, number_of_repeats):
    """Returns iterator where each item is repeated."""
    for item in iterator:
        for _ in range(number_of_repeats):
            yield item

def make_skip_and_repeat_iterator(iterator, number_of_skips, number_of_insertions):
    return make_iterator_with_repeats(make_skip_iterator(iterator, number_of_skips),
                               number_of_insertions)