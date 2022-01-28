import pandas as pd
import numpy as np

def count_total_rows(json_path, json_file, chunk_num):
    with open(json_path + json_file, "r") as f:
        reader = pd.read_json(json_path + json_file, lines=True, chunksize=chunk_num)
        size = 0
        for chunk in reader:
            size += chunk.shape[0]
    f.close()
    return size

def to_lower_case(string):
    try:
        return string.lower()
    except:
        pass