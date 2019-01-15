# This file downloads the necessary data files and the embeddings
# You need to have about 21 GB of free space to make this work
# The embedding zip is hosted on Google Drive by me, since it's not possible to download it directly from Keggle easily

# Should the automatic download of the data files fail, please download the files from of of these sources:
# https://www.kaggle.com/c/quora-insincere-questions-classification/download/test.csv
# https://www.kaggle.com/c/quora-insincere-questions-classification/download/train.csv
# In that case please place the files in the "input" folder and re-execute this script
#
# Should the automatic download of the embeddings fail, please download the file from of of these sources:
# https://www.kaggle.com/c/quora-insincere-questions-classification/download/embeddings.zip
# In that case please place the file in the "input" folder and re-execute this script

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

def missing_embeddings():
    if not os.path.isfile('input/embeddings/glove.840B.300d/glove.840B.300d.txt'):
        return True
    if not os.path.isfile('input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'):
        return True
    if not os.path.isfile('input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'):
        return True
    if not os.path.isfile('input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'):
        return True
    return False

if not os.path.isfile('input/train.csv.zip'):
    print("Downloading Training File")
    file_id = '12FX9Y7V_PssRO24myXVmn89uGwZ1zEsq'
    destination = 'input/train.csv.zip'
    download_file_from_google_drive(file_id, destination)

if not os.path.isfile('input/test.csv.zip'):
    print("Downloading Test File")
    file_id = '1_xCv-rYfXHYe1ZGjYfU2-C50ySwXCu78'
    destination = 'input/test.csv.zip'
    download_file_from_google_drive(file_id, destination)

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

if missing_embeddings():
    print("Unzipping Embeddings")
    with zipfile.ZipFile('input/embeddings.zip', 'r') as zip_ref:
        zip_ref.extractall("input/embeddings")

print("All necessary data is present. Good Job!")