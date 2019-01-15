# Kaggle Quora Challenge

## How to run this repo
1. Install the required packages listed in the dependencies by using "pip install [name]"
2. Execute the `data_download.py` file
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

## Requirements
- Have around 22 GB of free disk space
- Have 16GB+ of RAM for loading the embeddings
- Have all specified packages & data files
- Recommended: Have a GPU
- Recommended: Have tensorflow-gpu installed

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

## Rights
- All right reserved
- Certain code snippets are taken from different owners. I tried to include all sources.
- This GitHub Repo https://github.com/MarvinThiele/KaggleQuoraChallenge

## Troubleshooting
1. I receive GPU OOM errors when trying to execute the code
    - To fix this issue you can reduce the batch size in the code (at `train_pred()`)
    - If this doesnt fix the issue, it's most likely the embedding layer, which can not be reduced in size
    - Use a server with more VRAM in this case
2. My computer freezes and doesn't work anymore
    - This can happen if you are running less than 16GB of RAM
    - This is caused by loading the word embeddings into the memory
3. I have problems not listed here
    - Contact me at marvin.thiele@student.hpi.uni-potsdam.de

## Code from other authors
- I have taken code snippets & inspiration from other public kaggle notebooks
- Special Thanks to:
    - https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
    - https://www.kaggle.com/shujian/different-embeddings-with-attention-fork-fork
    - https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
    - https://www.kaggle.com/gmhost/gru-capsule
    - https://www.kaggle.com/vanshjatana/magic-numbers-is-all-you-need-0-692-lb-986394
   