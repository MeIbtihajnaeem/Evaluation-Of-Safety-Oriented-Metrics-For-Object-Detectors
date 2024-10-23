import json


class FileUtils:

    @staticmethod
    def save_tokens_to_json(validation_samples, results_path):
        """
        Collects tokens from the validation samples and saves them to a JSON file.

        Parameters:
        validation_samples (dict): Dictionary where values contain tokens to be aggregated.
        results_path (str): The directory path where the 'token_list.json' file will be saved.

        Returns:
        list: A list of all tokens from the validation samples.
        """
        list_token = []
        for key in validation_samples.keys():
            list_token.extend(validation_samples[key])

        with open(f"{results_path}/token_list.json", "w") as outfile:
            json.dump(validation_samples, outfile)

        return list_token
