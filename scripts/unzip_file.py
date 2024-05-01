#!/usr/bin/env python3

from zipfile import ZipFile

with ZipFile("../data/raw_analyst_ratings.csv.zip", "r") as f:
    f.extractall("../data")