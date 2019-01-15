# This file downloads the necessary data files and the embeddings
# You need to have about 21 GB of free space to make this work
# The embedding zip is hosted on Google Drive by me, since it's not possible to download it directly from Keggle easily

# Should the automatic download of the embeddings fail, please download the file from of of these sources:
# https://drive.google.com/open?id=1qV7MpMWHToQXqtQVxv3Ev6UAjJEM39f2
# https://www.kaggle.com/c/quora-insincere-questions-classification/download/embeddings.zip
#
# In that case please place the file in the "input" folder and reexecute this script for unzipping the file

import zipfile
import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if not os.path.isfile('input/test.csv'):
    print("Unzipping Test File")
    with zipfile.ZipFile('input/test.csv.zip', 'r') as zip_ref:
        zip_ref.extractall("input")

if not os.path.isfile('input/train.csv'):
    print("Unzipping Training File")
    with zipfile.ZipFile('input/train.csv.zip', 'r') as zip_ref:
        zip_ref.extractall("input")

if not os.path.isfile('input/embeddings.zip'):
    print("Downloading Embeddings")
    file_id = '1qV7MpMWHToQXqtQVxv3Ev6UAjJEM39f2'
    destination = 'input/embeddings.zip'
    download_file_from_google_drive(file_id, destination)

print("Unzipping Embeddings")
with zipfile.ZipFile('input/embeddings.zip', 'r') as zip_ref:
    zip_ref.extractall("input/embeddings")

print("All data is present and working. Good Job!")