#!/bin/bash

python3 scan.py $1
python3 fci.py $1
python3 delete_duplicate.py $1
python3 clean_duplicate_imgs.py $1
python3 make_fci_html.py $1
