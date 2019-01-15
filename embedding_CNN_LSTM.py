import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import re
import copy
from sklearn.metrics import f1_score

## some config values
embed_size = 300  # how big is each word vector
max_features = 95000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70  # max number of words in a question to use

def clean_text(x):
    x = str(x).lower()
    question = copy.deepcopy(x)
    to_remove = ['the', 'what', 'to', 'a', 'in', 'is', 'of', 'i', 'how', 'and', 'the']
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
    train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=1339)

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


train_X, val_X, test_X, train_y, val_y, word_index = load_and_prec(preprocess=True)
print(f"Word Index: {len(word_index)}")
print("Loaded Data")


def load_glove(word_index):
    max_features = len(word_index) + 1
    EMBEDDING_FILE = 'input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf8') if o.split(" ")[0] in word_index)

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


glove_matrix, glove_index = load_glove(word_index)
print("Loaded Gloves")


def train_pred(model, epochs=2, batch_size=1024):
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=batch_size, epochs=1, validation_data=(val_X, val_y))
        pred_val_y = model.predict([val_X], batch_size=batch_size, verbose=0)
    pred_test_y = model.predict([test_X], batch_size=batch_size, verbose=0)
    return pred_val_y, pred_test_y


def model_cnn(embedding_matrix, word_index):
    embed_size = 300

    inp = Input(shape=(maxlen,))
    x = Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    conv1 = Conv2D(64, kernel_size=(1, embed_size), kernel_initializer='he_normal', activation='elu')(x)
    mp1 = MaxPool2D(pool_size=(70, 1))(conv1)

    conv2 = Conv2D(64, kernel_size=(3, embed_size), kernel_initializer='he_normal', activation='elu')(x)
    mp2 = MaxPool2D(pool_size=(68, 1))(conv2)

    conv3 = Conv2D(64, kernel_size=(5, embed_size), kernel_initializer='he_normal', activation='elu')(x)
    mp3 = MaxPool2D(pool_size=(66, 1))(conv3)

    mp_layers = [mp1, mp2, mp3]

    z = Concatenate(axis=1)(mp_layers)
    z = Flatten()(z)
    z = Dropout(0.3)(z)

    outp = Dense(60, activation="relu")(z)
    outp = Dense(1, activation="sigmoid")(outp)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def model_lstm(embedding_matrix, word_index):
    max_features = len(word_index) + 1

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    p1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    p1 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(p1)

    p2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    p2 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(p2)

    p3 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    p3 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(p3)

    lstm_layers = [p1, p2, p3]

    y = Concatenate(axis=1)(lstm_layers)
    y = Flatten()(y)

    outp = Dense(64, activation="relu")(y)
    outp = Dense(1, activation="sigmoid")(y)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Checkout Layers
# example_cnn = model_dnn(glove_matrix, word_index)
# example_cnn.layers
# for layer in example_cnn.layers:
#     print(layer)
#     print(layer.input_shape)
#     print(layer.output_shape)

outputs = []
pred_val_y, _ = train_pred(model_cnn(glove_matrix, word_index), epochs = 2, batch_size=512)
thresh = 0.05
f1_scores = []
while thresh <= 1:
    pred_val_y_decided = (pred_val_y > thresh).astype(int)
    thresh += 0.05
    f1_scores.append(f1_score(val_y, pred_val_y_decided))
outputs.append(max(f1_scores))

for max_score in outputs:
    print(f"Max F1 for CNN Score is: {max_score}")

outputs = []
pred_val_y, _ = train_pred(model_lstm(glove_matrix, word_index), epochs = 2, batch_size=512)
thresh = 0.05
f1_scores = []
while thresh <= 1:
    pred_val_y_decided = (pred_val_y > thresh).astype(int)
    thresh += 0.05
    f1_scores.append(f1_score(val_y, pred_val_y_decided))
outputs.append(max(f1_scores))

for max_score in outputs:
    print(f"Max F1 for LSTM Score is: {max_score}")