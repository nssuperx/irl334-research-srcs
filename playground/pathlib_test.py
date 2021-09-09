import pathlib
import sys

args = sys.argv
input_filepath: str = args[1]

p = pathlib.PureWindowsPath(input_filepath)
print(p.name)
print(p.stem)
print(p.suffix)
print(p.parent)
print(p.with_suffix(".txt"))

outfilename = pathlib.Path(p.name)
outfilepath = outfilename.with_suffix(".txt")

print(outfilepath)
