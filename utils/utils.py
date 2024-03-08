def print_map(map_array):
    for row in map_array:
        print(row)
    print("\n")


def get_merge_dictionary(dict1: dict, dict2: dict):
    """
    This function returns a merged dictionary
    adding keys of dict1 and values of dict2
    :param dict1:
    :param dict2:
    :return: dictionary merged keys of dict1 and values of dict2
    """
    result = {}
    for key in dict1:
        result[key] = dict2[key]
    return result
