import os
import time
import urllib.request

def download_if_necessary(url, input_filepath, count_max=100):
    if not os.path.exists(input_filepath):
        with urllib.request.urlopen(url) as response, open(input_filepath, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        count = 1
        while not os.path.exists(input_filepath) and (count != count_max):
            time.sleep(1)
            count = count + 1