import os
import urllib.request as request
from zipfile import ZipFile
from deepClassifier.entity import DataIngestionConfig
from deepClassifier import logging
from deepClassifier.utils import *
from tqdm import tqdm


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        logging.info("Trying to download the file")
        if not os.path.exists(self.config.local_data_file):
            logging.info("Download started")
            filename, header = request.urlretrieve(
                url=self.config.source_URL, filename=self.config.local_data_file
            )
            logging.info(
                f"File : {filename} downloaded successfuly and the header is {header}"
            )
        else:
            logging.info(
                f"File already exist of size {get_size(Path(self.config.local_data_file))}"
            )

    def _get_updated_list_of_file(self, list_of_file):
        return [
            f for f in list_of_file if f.endswith(".jpg") and ("Cat" in f or "Dog" in f)
        ]

    def _preprocess(self, zf: ZipFile, f: str, working_dir: str):
        target_filepath = os.path.join(working_dir, f)
        if not os.path.exists(target_filepath):
            # Extract a single file from zip
            zf.extract(f, working_dir)

        if os.path.getsize(target_filepath) == 0:
            logging.info(
                f"Removing file : {target_filepath} of size {get_size(Path(target_filepath))}"
            )
            os.remove(target_filepath)

    def unzip_and_clean(self):
        with ZipFile(file=self.config.local_data_file, mode="r") as zf:
            list_of_files = zf.namelist()
            updated_list_of_file = self._get_updated_list_of_file(list_of_files)
            for f in tqdm(updated_list_of_file):
                self._preprocess(zf, f, self.config.unzip_dir)
