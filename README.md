# Kaggle Quora Challenge

## How to run this repo
1. Install the required packages listed in the dependencies by using "pip install [name]"
2. Execute the data_download.py file
    - Make sure you see the success message at the end
    - Should the file download fail, follow the instruction at the top of the file
3. Execute any file you want!
    - Each file is completely self contained, since these files used to be notebooks on kaggle.com
    - You can find a description below
## Available Approaches & Descriptions
- `data_analysis.py`   Gives an overview over the data
- `embedding_CNN_LSTM.py` Contains a CNN and a LSTM classifier
- `meanembedding_SVM` Contains a SVM classifier with mean embedding vectorization
- `metaembedding_Capsule` Contains a Capsule classifier with a meta embedding (External)
- `metaembedding_Ensemble` Contains a Ensemble classifier with a meta embedding (External)
- `preprocessing_comparison` Contains a comparison between using and not using preprocessing before the embedding
- `tfidf_SVM` Contains a SVM classifier with TF-IDF vectorization

## Dependencies
Please make sure to have these python packages installed before runnging the code
```python
pandas
tqdm
textblob
timeit
sklearn
keras
tensorflow
numpy
```

## Requirements
- Have around 22 GB of free disk space
- Have 16GB+ of RAM for loading the embeddings
- Optional: Have tensorflow-gpu installed

## Rights
- All right reserved
- Certain code snippets are taken from different owners. I tried to include all sources.