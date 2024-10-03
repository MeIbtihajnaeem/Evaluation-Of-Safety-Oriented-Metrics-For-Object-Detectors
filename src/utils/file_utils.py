
import json
import os

class FileUtils:
    def write_data_to_file(self, output_folder_path, key, value):
        if os.path.exists(output_folder_path):
            # If the file exists, load the existing content
            with open(output_folder_path, 'r') as f:
                data = json.load(f)
        else:
            # If the file doesn't exist, create an empty dictionary
            data = {}

        # Update the dictionary with the new data
        data[key] = value

        # Write the updated data back to the file
        with open(output_folder_path, 'w') as f:
            json.dump(data, f, indent=4)

    def read_data_from_file(self, save_in_path):
        with open(save_in_path, "r") as f:
            data = json.load(f)
        return data
