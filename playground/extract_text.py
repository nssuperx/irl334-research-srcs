from pdfminer.high_level import extract_text
import sys
import pathlib

args = sys.argv
input_filepath: str = args[1]

text = extract_text(input_filepath)

p = pathlib.Path(input_filepath)
out_filepath = p.with_suffix(".txt")

with open(out_filepath, mode='w', encoding="utf-8") as f:
    f.write(text)
