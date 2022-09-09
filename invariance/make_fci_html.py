# 参考: https://qiita.com/nshinya/items/a46ef0002284d2f77789

import sys
import json
import pandas as pd
from modules.io import FciDataManager

html_template = """
<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">

    <title>fci</title>
  </head>
  <body>
    <!-- Optional JavaScript; choose one of the two! -->

    <!-- Option 1: Bootstrap Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM" crossorigin="anonymous"></script>
    <div class="container">
        {table}
    </div>
  </body>
</html>
"""

default_dataset = 1
args = sys.argv


def main():
    if (len(args) >= 2):
        dataset_number = args[1]
    else:
        dataset_number = default_dataset
    dataMgr: FciDataManager = FciDataManager(dataset_number)
    df = pd.read_csv(f"{dataMgr.out_dirpath}/crf_cluster.csv", index_col=0)
    with open(f"{dataMgr.dirpath}/setting.json", "r") as f:
        setting = json.load(f)
    crf_width = int(setting["default"]["crf_width"])
    df["image"] = df.index.map(lambda s: f'<img src="./crf-clean/{s}.png" width="{crf_width}" loading="lazy" />')
    table = df.to_html(classes=["table", "table-bordered", "table-hover"], escape=False, show_dimensions=True)
    html = html_template.format(table=table)
    html = html.replace('<thead>', '<thead class="sticky-top bg-white">')

    with open(f"{dataMgr.out_dirpath}/fci-tables.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
