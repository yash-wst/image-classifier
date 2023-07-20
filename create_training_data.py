#!/usr/bin/env python3

import os
import shutil
import pathlib
import datetime
import logging
import requests
import sys
from zipfile import ZipFile


BASE_PATH = "training_dataset/good/"

logging.basicConfig(
    filename="./training_data.log",
    level=logging.DEBUG,
    encoding="utf-8",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def download_and_unzip(url):
    name = str(url).split("/")[-1]
    logging.info(f"Downloading {name}")
    print(f'Downloading {name}')
    with requests.get(url=url, headers={"Referer": "https://osm.vnsguapps.in/"}) as r:
        with open(f"{name}", "wb") as dest:
            dest.write(r.content)
        logging.info(f"Downloaded file {name}")
        with ZipFile(name, 'r') as zipfile:
            zipfile.extractall(BASE_PATH+name.split('.')[0])
        logging.info(f"Extracted zipfile to {BASE_PATH+name.split('.')[0]}")


def iterator_tune(path):
    logging.info(f"Iterating over {path}")
    p = pathlib.Path(path)
    for item in p.iterdir():
        if item.is_dir():
            logging.info(f"{item} is a directory.")
            iterator_tune(item)
        # If item is a file and if item is not in the root directory
        elif item.is_file() and pathlib.Path(item) != pathlib.Path(
            BASE_PATH + os.path.basename(item).split("/")[-1]
        ):
            # If item already exists in the root directory, rename it with datetime suffix
            if os.path.exists(BASE_PATH + os.path.basename(item).split("/")[-1]):
                logging.debug(
                    f"{item} already exists in root training directory. Renaming and moving."
                )
                os.rename(item, str(item) + str(datetime.datetime.now()))
            shutil.move(item, BASE_PATH)

    if path != BASE_PATH:
        logging.info(f"Deleting directory {path}")
        os.rmdir(path)


if __name__ == "__main__":
    # print(sys.argv[1])
    download_and_unzip(sys.argv[1])
    iterator_tune(BASE_PATH)
