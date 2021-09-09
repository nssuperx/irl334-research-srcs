from pdfminer.high_level import extract_text
import sys
import pathlib
import os

args = sys.argv
input_filepath: str = args[1]

text = extract_text(input_filepath)

# out_filepath = 

with open(path_w, mode='w') as f:
    f.write(s)
