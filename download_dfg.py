from zipfile import ZipFile
import boto3
import yaml
import os
from os import path
import tarfile

if path.exists('./data/dfg/JPEGImages'):
    print('files already downloaded')
    exit()

if not path.exists('./data/dfg'):
    os.makedirs('./data/dfg')

with open("./s3.yaml", "r") as f:
    config = yaml.safe_load(f)
session = boto3.session.Session()
client = session.client("s3", **config["s3"])

client.download_file("dfg", 'JPEGImages.tar.bz2', path.join('./data/dfg',  'JPEGImages.tar.bz2'))
client.download_file("dfg", 'DFG-tsd-aug-annot-json.zip', path.join('./data/dfg',  'DFG-tsd-aug-annot-json.zip'))

with tarfile.open('./data/dfg/JPEGImages.tar.bz2', 'r:bz2') as tar:
    tar.extractall()


with ZipFile(f"./data/dfg/DFG-tsd-aug-annot-json.zip", "r") as annot_zip:
    annot_zip.extractall(f"./data/dfg")
