import os


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


def make_model_name(hyperparameters: dict):
    """
    This function takes a dictionary of hyperparameters and returns a string
    :param hyperparameters:
    :return:
    """
    name = ""
    for key in hyperparameters:
        name += f"{simplify_key(key)}_{hyperparameters[key]}_"
    return name


def make_model_directory(input_object: dict):
    dir_path = f"models/${input_object['map_name']}/${input_object['algorithm']}"
    if not os.path.join(dir_path):
        os.makedirs(dir_path)


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_input(file_path: str):
    file = open(file_path, 'r')
    content = file.read()
    file.close()
    print(content.split('\n'))
    input_object = {}
    for i in content.split('\n'):
        if i and i[0] != '#':
            key, value = i.split(' : ')
            input_object[key] = value

    return input_object
