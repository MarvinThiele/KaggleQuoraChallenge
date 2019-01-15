import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import re
import copy
import operator
from textblob import TextBlob
from timeit import default_timer as timer

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

    for word in to_remove:
        split = x.split(" ")
        if word in split:
            x = re.sub(" " + word + " ", ' ', x)
            x = re.sub(word + "$", ' ', x)
            x = re.sub("^" + word, ' ', x)

    for word in mispell_dict.keys():
        split = x.split(" ")
        if word in split:
            x = re.sub(" " + word + " ", " " + mispell_dict[word] + " ", x)
            x = re.sub(word + "$", " " + mispell_dict[word] + " ", x)
            x = re.sub("^" + word, " " + mispell_dict[word] + " ", x)

    return x

train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
train_df_reduced = train_df.loc[0:100000]

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

train_df_list = train_df["question_text"].str.lower().tolist()

# Calculate the most common words
cust_word_index = {}
for question in train_df_list:
    for word in question.split(" "):
        if word in cust_word_index.keys():
            cust_word_index[word] = cust_word_index[word]+1
        else:
            cust_word_index[word] = 1
len(cust_word_index)

sorted_word_index = sorted(cust_word_index.items(), key=operator.itemgetter(1), reverse=True)
print("Finished")

# Calculate the average length of the sentences
sentence_sum = 0
length_array = []
for question in train_df_list:
    sentence_sum += len(question.split(" "))
    length_array.append(len(question.split(" ")))
sentence_avg = sentence_sum / len(train_df_list)
print(f"The average sentence length is: {sentence_avg}")
length_array = sorted(length_array)
median = length_array[int(len(length_array)/2)]
print(f"The median sentence length is: {median}")

# Misspelling correction
start = timer()
train_df["question_text"] = train_df['question_text'].apply(lambda x: str(TextBlob(x).correct()))
end = timer()
print(end - start)

# Misspelling correction
start = timer()
train_df_reduced['corr_question_text'] = train_df_reduced['question_text'].apply(lambda x: str(TextBlob(x).correct()))
end = timer()
print(end - start)

# Sentiment analysis
start = timer()
train_df_reduced['sentiment'] = train_df_reduced['question_text'].apply(lambda x: TextBlob(x).sentiment[0] )
end = timer()
print(end - start)

train_df_reduced[:15]

