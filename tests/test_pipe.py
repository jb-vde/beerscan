import py_compile
import pandas as pd
import numpy as np
import os
from beerscan.model.pipeline import main_pipe
import cv2


result_df = pd.DataFrame(columns=["real_name","predicted_name"])

# assign directory
directory = 'tests/testfile/images'

# iterate over files in
# that directory

for filename in os.listdir(directory):
    f = cv2.imread(os.path.join(directory, filename))
    data = main_pipe(f)
    result_df.loc[len(result_df)] = [filename,[elem["beer_name"] for elem in data.values()]]

result_df.to_csv("tests/testfile/csv/results")
