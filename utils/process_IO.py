def model_name(input_object: dict):
    if input_object['model_name'] == '':
        return f"{input_object['step']}_{input_object['learning_rate']}"
    else:
        return input_object['model_name']


def make_model_directory(input_object: dict):
    dir_path = f"models/${input_object['map_name']}/${input_object['algorithm']}"
    import os
    if not os.path.join(dir_path):
        os.makedirs(dir_path)


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
