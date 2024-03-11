def load_map_name(map_dir: str):
    """
    This function is used to load the maps from the given directory
    :param map_dir: path
    :return: list of maps
    """
    import os
    return os.listdir(map_dir)
