def load_map_name(map_dir: str):
    """
    This function is used to load the maps from the given directory
    :param map_dir: directory path
    :return: list of maps
    """
    import os
    maps = os.listdir(map_dir)
