import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.callbacks import *
from tqdm import tqdm
tqdm.pandas()

## some config values
max_features = 95000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70  # max number of words in a question to use


def clean_text(x):
    x = str(x).lower()
    mispell_dict = {'colour': 'color',
                    'centre': 'center',
                    'favourite': 'favorite',
                    'travelling': 'traveling',
                    'counselling': 'counseling',
                    'theatre': 'theater',
                    'cancelled': 'canceled',
                    'labour': 'labor',
                    'organisation': 'organization',
                    'wwii': 'world war 2',
                    'citicise': 'criticize',
                    'cryptocurrencies': 'crypto currencies',
                    'ethereum': 'crypto currency',
                    'coinbase': 'crypto market',
                    'altcoins': 'crypto currency',
                    'litecoin': 'crypto currency',
                    'altcoin': 'alt coin'}

    #
    # Clean punctuations
    for punct in "/-":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%()*+-/:;<=>@[\\]^_`{|}~' + '“”':
        x = x.replace(punct, '')
    for punct in '’\'`´•~£·_©^®<→°€™›♥←×§″′█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√₹':
        x = x.replace(punct, '')

    for word in mispell_dict.keys():
        split = x.split(" ")
        if word in split:
            x = re.sub(" " + word + " ", " " + mispell_dict[word] + " ", x)
            x = re.sub(word + "$", " " + mispell_dict[word] + " ", x)
            x = re.sub("^" + word, " " + mispell_dict[word] + " ", x)

    return x


def load_glove_index(word_index):
    EMBEDDING_FILE = 'input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embedding_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8") if o.split(" ")[0] in word_index)
    to_del = []
    for key in embedding_index.keys():
        if len(embedding_index[key]) < 300:
            to_del.append(key)
    for key in to_del:
        del embedding_index[key]
    return embedding_index


def pad_strings(input_array, maxlen):
    padded_strings = []
    for sentence in input_array:
        sentence = sentence.split(" ")
        while len(sentence) < maxlen:
            sentence.append("")
        padded_strings.append(sentence[0:maxlen])
    return padded_strings


def mean_vectorize(data, embedding_index):
    vec_data = []
    for data_point in data:
        word_vecs = []
        # Transform the words
        for word in data_point:
            if word in embedding_index:
                word_vecs.append(embedding_index[word])
            else:
                continue
        if len(word_vecs) == 0:
            vec_data.append(np.zeros(300))
        else:
            # Average them
            vec_data.append(np.average(word_vecs, axis=0))
    return vec_data


train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=1341)

train_X = train_df["question_text"].values
val_X = val_df["question_text"].values
test_X = test_df["question_text"].values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))

train_X = pad_strings(train_X, maxlen)
val_X = pad_strings(val_X, maxlen)
test_X = pad_strings(test_X, maxlen)

glove_index = load_glove_index(tokenizer.word_index)

train_X = mean_vectorize(train_X, glove_index)
val_X = mean_vectorize(val_X, glove_index)
test_X = mean_vectorize(test_X, glove_index)

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values

print("Done!")

# SVM
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, class_weight={1: 20})
clf.fit(train_X, train_y)

print(metrics.classification_report(val_y, clf.predict(val_X), target_names=["Non-Toxic", "Toxic"]))
print(f"Combiend F1-Score: {metrics.f1_score(val_y, clf.predict(val_X))}")