import pathlib

input_filepath: str = r'd:\Users\test\test.pdf'

print(input_filepath)

p = pathlib.PureWindowsPath(input_filepath)
print(p.name)
print(p.stem)
print(p.suffix)
print(p.parent)
print(p.with_suffix(".txt").as_posix())

outfilename = pathlib.Path(p.name)
outfilepath = outfilename.with_suffix(".txt")

print(outfilepath)
