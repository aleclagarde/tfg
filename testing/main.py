import requests
import json
import os
import numpy

RESULTS_PATH = "results"


class MakeApiCall:
    def formatted_print(self, obj):
        text = json.dumps(obj, sort_keys=True, indent=4)
        print(text)

    def get_results(self, api, latencies):
        url = api + "/results"
        response = requests.get(url)
        folder_name = f'T5_'
        if response.status_code == 200:
            print("successfully fetched the data")
            output_path = os.path.join(RESULTS_PATH, folder_name)
            if not (os.path.isdir(output_path)):
                os.makedirs(output_path)
            # Save codecarbon logs
            with open(os.path.join(output_path, "API_results.csv"), "wb") as file:
                file.write(response.content)

            # Save latencies into csv
            numpy.savetxt(os.path.join(output_path, "latency_results.csv"), latencies, delimiter=",", header="latency")
        else:
            print(f"Hello person, there's a {response.status_code} error with your request")

    def make_requests(self, api, sentences):
        url = api + "/invocations"
        requests_latency = []
        it = 0
        for sentence in sentences:
            my_obj = {"language": "German", "text": sentence}
            response = requests.post(url, json=my_obj)
            if response.status_code == 200:
                print("successfully fetched the data with parameters provided")
                self.formatted_print(response.json())
                requests_latency.append(response.elapsed.total_seconds())
                print("Latency in seconds of the request[" + str(it) + "]: " + str(response.elapsed.total_seconds()))
            else:
                print(f"Hello person, there's a {response.status_code} error with your request")
            it = it + 1

        return requests_latency

    def __init__(self):

        with open("sentences.txt") as my_file:
            sentences_to_post = my_file.read().splitlines()
        with open("target_sentences.txt") as my_file:
            target_sentences = my_file.read().splitlines()

        provider = ["https://tfgdemo2.azurewebsites.net", "azure"]
        print("Requesting provider: " + provider[1])
        latencies = self.make_requests(provider[0], sentences_to_post)
        self.get_results(provider[0], latencies)


if __name__ == "__main__":
    api_call = MakeApiCall()
