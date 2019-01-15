import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import re
import operator
from textblob import TextBlob
from timeit import default_timer as timer

train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
train_df_reduced = train_df.loc[0:10]

train_df_list = train_df["question_text"].str.lower().tolist()

# Calculate the most common words
cust_word_index = {}
for question in train_df_list:
    for word in question.split(" "):
        if word in cust_word_index.keys():
            cust_word_index[word] = cust_word_index[word]+1
        else:
            cust_word_index[word] = 1
print(f"Number of unique words in training set: {len(cust_word_index)}")

sorted_word_index = sorted(cust_word_index.items(), key=operator.itemgetter(1), reverse=True)
print("Most common words:")
print(sorted_word_index[0:10])

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
train_df_reduced["question_text_corrected"] = train_df_reduced['question_text'].apply(lambda x: str(TextBlob(x).correct()))
end = timer()
print(f"Sample Misspelling Correction took {end - start} seconds")

# Sentiment analysis
start = timer()
train_df_reduced['sentiment'] = train_df_reduced['question_text'].apply(lambda x: TextBlob(x).sentiment[0] )
end = timer()
print(f"Sample Sentiment Analysis took {end - start} seconds")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(train_df_reduced)