import itertools


a_set = {"a", "b", 1, 2}
data = itertools.combinations(a_set, 2)
subsets = list(data)
print (subsets)