import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import gc
import re
import copy
import operator

from sklearn.metrics import f1_score

## some config values
embed_size = 300  # how big is each word vector
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


def load_and_prec(preprocess=True):
    train_df = pd.read_csv("input/train.csv")
    test_df = pd.read_csv("input/test.csv")
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

    if preprocess:
        train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
        test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

    ## split to train and val
    train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=5620)

    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    val_X = val_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    print(len(tokenizer.word_index))
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences
    train_X = pad_sequences(train_X, maxlen=maxlen)
    val_X = pad_sequences(val_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values
    val_y = val_df['target'].values

    # shuffling the data
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))
    val_idx = np.random.permutation(len(val_X))

    train_X = train_X[trn_idx]
    val_X = val_X[val_idx]
    train_y = train_y[trn_idx]
    val_y = val_y[val_idx]

    return train_X, val_X, test_X, train_y, val_y, tokenizer.word_index


data_array = []
embeddings = []

# Compare the length of the word indexes with and without preprocessing
train_X, val_X, test_X, train_y, val_y, word_index = load_and_prec(preprocess=False)
print(f"Native Word Index Count: {len(word_index)}")
data_array.append(
    [copy.deepcopy(train_X), copy.deepcopy(val_X), copy.deepcopy(test_X), copy.deepcopy(train_y), copy.deepcopy(val_y),
     copy.deepcopy(word_index)])

train_X, val_X, test_X, train_y, val_y, word_index = load_and_prec(preprocess=True)
print(f"Preprocessed Word Index Count: {len(word_index)}")
data_array.append(
    [copy.deepcopy(train_X), copy.deepcopy(val_X), copy.deepcopy(test_X), copy.deepcopy(train_y), copy.deepcopy(val_y),
     copy.deepcopy(word_index)])
print("Loaded Data")

del train_X, val_X, test_X, train_y, val_y, word_index
gc.collect()


def load_glove(word_index):
    max_features = len(word_index) + 1
    EMBEDDING_FILE = 'input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8") if o.split(" ")[0] in word_index)

    to_del = []
    for key in embeddings_index.keys():
        if len(embeddings_index[key]) < 300:
            to_del.append(key)
    for key in to_del:
        del embeddings_index[key]

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return (embedding_matrix, embeddings_index)


embeddings.append(load_glove(data_array[0][5]))
embeddings.append(load_glove(data_array[1][5]))
print("Loaded Gloves")

# Rereading data-files for some analysis with embeddings
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")
train_df_list = train_df["question_text"].tolist()

# Find and print coverage:
for i in range(len(data_array)):
    word_index = data_array[i][5]
    embedding_index = embeddings[i][1]
    covered_words = 0
    not_covered_words = []

    for word in word_index:
        if word in embedding_index:
            covered_words += 1
        else:
            not_covered_words.append(word)

    print(f"We had {covered_words} / {len(word_index)}")

word_index_large = data_array[0][5]
word_index_procc = data_array[1][5]

not_emb_words = []
for word in word_index_procc:
    if word in embeddings[1][1]:
        continue
    else:
        not_emb_words.append(word)

cust_word_index = {}
for question in train_df_list:
    for word in question.split(" "):
        if word in cust_word_index.keys():
            cust_word_index[word] = cust_word_index[word]+1
        else:
            cust_word_index[word] = 1

not_emb_count_dict = {}
for word in not_emb_words:
    if word in cust_word_index:
        not_emb_count_dict[word] = cust_word_index[word]

sorted_emb_index = sorted(not_emb_count_dict.items(), key=operator.itemgetter(1), reverse=True)
print("Top words which are not covered by the embedding but present in the data: ")
print(sorted_emb_index[0:20])

def model_cnn(embedding_matrix, word_index):
    max_features = len(word_index) + 1
    filter_sizes = [1, 2, 3, 5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                      kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_pred(model, data, epochs=2):
    train_X = data[0]
    val_X = data[1]
    test_X = data[2]
    train_y = data[3]
    val_y = data[4]
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    return pred_val_y, pred_test_y

outputs = []
for i in range(len(embeddings)):
    pred_val_y, _ = train_pred(model_cnn(embeddings[i][0], data_array[i][5]), data_array[i], epochs=2)
    thresh = 0.05
    f1_scores = []
    while thresh <= 1:
        pred_val_y_decided = (pred_val_y > thresh).astype(int)
        thresh += 0.05
        f1_scores.append(f1_score(data_array[i][4], pred_val_y_decided))
    outputs.append(max(f1_scores))

for i, max_score in enumerate(outputs):
    if i == 0:
        print(f"Native Max F1 Score is: {max_score}")
    else:
        print(f"Preprocessed Max F1 Score is: {max_score}")