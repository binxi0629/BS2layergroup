import numpy as np


def crystal_number(s: int):
    for c, m in enumerate([2, 15, 74, 142, 167, 194, 230]):
        if s <= m:
            return c + 1


def spacegroup_index_lower(c: int):
    margins = [2, 15, 74, 142, 167, 194, 230]
    lower = margins[c - 2] if c > 1 else 0
    return lower


def spacegroup_index_upper(c: int):
    margins = [2, 15, 74, 142, 167, 194, 230]
    upper = margins[c - 1]
    return upper


def spacegroup_number_range(c: int):
    return range(spacegroup_index_lower(c) + 1, spacegroup_index_upper(c) + 1)


def new_category(na_group_list:list):
    spacegroup_list = []
    for i in range(230):
        if i+1 in na_group_list:
            pass
        else:
            spacegroup_list.append(i+1)

    spacegroup_list.append(231)
    return np.array(spacegroup_list), len(spacegroup_list)


def crystal_system_sizes():
    return [2, 15, 74, 142, 167, 194, 230]