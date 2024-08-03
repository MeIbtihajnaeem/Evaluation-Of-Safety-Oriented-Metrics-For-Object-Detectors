import json


class FileUtils:
    def write_data_to_file(self, output_folder_path, key, value):
        with open(output_folder_path, 'w') as f:
            json.dump({key: value}, f)

    def read_data_from_file(self, save_in_path):
        f = open(save_in_path, "r")
        data = json.load(f)
        return data
