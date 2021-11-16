import numpy as np
from pyxtal import pyxtal, symmetry, operations


def apply_pbc(pos_list):  # 1x3 array

    def apply_pbc_once(pos):
        if pos > 1:
            pos -= 1
        elif pos < 0:
            pos += 1
        else:
            return pos
        return apply_pbc_once(pos)

    for i in range(3):
        pos_list[i] = apply_pbc_once(pos_list[i])

    return pos_list


def check_sys_operation_valid(pos_after_op:np.array, element:int, pos_list:np.array, elements_list,tolerance=0.005)->bool:
    """

    :param pos_after_op: positions after applying syymetry operations
    :param element: input element type
    :param pos_list: positions of the total atoms
    :param elements_list: atoms type of the cell
    :param tolerance:
    :return:
    """
    # (pos_list.shape)
    num_op = pos_after_op.shape[0]
    num_elem = len(elements_list)
    count = 0
    for p in range(num_elem):
        if elements_list[p] == element:
            # print(pos_list[p, :])
            # print(elements_list[p])
            valid_list = np.sum(np.array(np.absolute(pos_after_op - pos_list[p,:]) <= tolerance), axis=1, keepdims=False) # num_opx1
            # print(valid_list)

            for i in valid_list:
                if 3 == i:
                    count += 1
    if num_op == count:
        return True
    else:
        return False


def check_layergroup(pos, elements_list,layergroup_num):
    """

    :param pos: fractional positions of atoms [nx3] where n is number of atoms
    :param elements_list: atomic number list of atoms [nx1]
    :param layergroup_num: layer group number: 1-80
    :return: bool: if it belongs to this layer group, if Ture: layer group number else: -1
    """
    opers = symmetry.get_layer_generators(layergroup_num)[0]
    num_elem = len(elements_list)
    result = []
    for i in range(num_elem):

        res = operations.apply_ops(pos[i, :], opers)

        # pbc_list = PBC([0,0,1])
        # pbc_list_pbc,amount=apply_pbc(res,pbc_list)
        for idx in range(res.shape[0]):
            res[idx, :] = apply_pbc(res[idx, :])

        result.append(check_sys_operation_valid(res, elements_list[i], pos, elements_list,tolerance=0.005))

    # print(result)
    if (False in result):
        # print("Not this layer group, try next")
        return False, -1
    else:
        # print("Found the layer group")
        return True, layergroup_num


def exhasutive_search_layergroup(pos, elements_list):
    """

    :param pos: fractional positions of atoms [nx3] where n is number of atoms
    :param elements_list:  atomic number list of atoms [nx1]
    :return: layergroup_num: layergroup number
    """
    for i in range(1,81):

        num = 81-i
        judgement, layergroup_num = check_layergroup(pos, elements_list, num)
        if judgement:
            break

    return layergroup_num

