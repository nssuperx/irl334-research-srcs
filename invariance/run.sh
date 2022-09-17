#!/bin/bash

python scan.py $1
python fci.py $1
python delete_duplicate.py $1
python clean_duplicate_imgs.py $1
python make_fci_html.py $1
