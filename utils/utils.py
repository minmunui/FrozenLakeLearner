def current_time_for_file():
    """
    This function returns the current time
    :return: current time
    """
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "_").replace(":", "_")

def print_map(map_array):
    for row in map_array:
        print(row)
    print("\n")


def simplify_key(key: str):
    """
    This function takes a string and returns a simplified version of the string
    :param key:
    :return:
    """
    key = key.lower()
    if "_" in key:
        temp = key.split("_")
        result = ""
        for i in temp:
            result += i[0]
        return result
    else:
        if len(key) > 1:
            return key[0:1]
        else:
            return key


def get_merge_dictionary(dict1: dict, dict2: dict):
    """
    This function returns a merged dictionary
    adding keys of dict1 and values of dict2
    :param dict1:
    :param dict2:
    :return: dictionary merged keys of dict1 and values of dict2
    """
    result = dict1
    for key in dict2:
        result[key] = dict2[key]
    return result
