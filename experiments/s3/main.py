import boto3
import boto3.session
import yaml

config_file = './s3.yaml'

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

session = boto3.session.Session()
client = session.client('s3', **config['s3'])

client.download_file('data', 'annotations.zip', './data/annotations-downloaded.zip')
