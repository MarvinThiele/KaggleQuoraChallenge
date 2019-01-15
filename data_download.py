# import urllib.request
# import zipfile
#
# test_csv_url = 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/10737/165023/test.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1547821731&Signature=skpmhXUBpuxZXN8q5sNeLvBCJunE7bQsMOU%2FPsYqvymryt%2FtYpzVChMgjgEhty4E1h%2Fl%2Bc01AUaXDrgTGWJaVClaF%2FX7jrsN9yvU%2BWj%2F07WlXgUNs3irR%2FWp%2BS64NHRHTLZD92nLi89lES%2FigawYAMZfleqYeM93Ms5j6FewmSPepjcaVU1sHwPBeuNuD6aM2c4%2BzKICZ%2BNvMT0mvfs3tc5Mrakmace%2BKCHypxj%2FvPN%2FN4g1DfkmyBp%2BcT%2FDEov69s53juvK5gDj69NKmy3cReglal19IbQXsEc6BUwZvQpBsrrNYHRRxGBheFop2LPMkFF1pbzuDDy%2FBWYw33plmA%3D%3D'
# train_csv_url = 'https://www.kaggle.com/c/quora-insincere-questions-classification/download/train.csv'
# embeddings_url = 'https://www.kaggle.com/c/quora-insincere-questions-classification/download/embeddings.zip'
#
# print('Downloading test.csv')
# urllib.request.urlretrieve(test_csv_url, 'input/test.csv')
#
# print('Downloading train.csv')
# urllib.request.urlretrieve(train_csv_url, 'input/train.csv')
#
# print("Downloading Embeddings")
# urllib.request.urlretrieve(embeddings_url, 'input/embeddings.zip')
#
# print("Unzipping Embeddings")
# with zipfile.ZipFile('input/embeddings.zip', 'r') as zip_ref:
#     zip_ref.extractall("input")
