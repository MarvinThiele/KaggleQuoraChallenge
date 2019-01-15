import pandas as pd
import copy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics


def balancedTrainingSet(dataframe):
    positiveExamples = dataframe.loc[dataframe['target'] == 1]
    negativeExamples = dataframe.loc[dataframe['target'] == 0]
    balancedSet = pd.concat([positiveExamples, negativeExamples[0:len(positiveExamples)]])
    balancedSet = balancedSet.sample(frac=1)
    return balancedSet


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

    for word in mispell_dict.keys():
        split = x.split(" ")
        if word in split:
            x = re.sub(" " + word + " ", " " + mispell_dict[word] + " ", x)
            x = re.sub(word + "$", " " + mispell_dict[word] + " ", x)
            x = re.sub("^" + word, " " + mispell_dict[word] + " ", x)

    return x


train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=1341)

print("Positive Exmaples: " + str(train_df.groupby('target')['qid'].nunique()[0]))
print("Negative Exmaples: " + str(train_df.groupby('target')['qid'].nunique()[1]))

train_X = train_df[['qid', 'question_text']]
train_Y = train_df[['target']]

val_X = val_df[['qid', 'question_text']]
val_Y = val_df[['target']]

test_X = test_df[['question_text']]

tfidf_vec = TfidfVectorizer()
transformedQuestions = tfidf_vec.fit_transform(train_X['question_text'])
transformedTestQuestions = tfidf_vec.transform(test_X['question_text'])
transformedValQuestions = tfidf_vec.transform(val_X['question_text'])

print("Classifying!")
clf = SGDClassifier(max_iter=1000, tol=1e-3, class_weight={1: 20})
clf.fit(transformedQuestions, train_Y)
output = clf.predict(transformedTestQuestions)

print(metrics.classification_report(val_Y, clf.predict(transformedValQuestions), target_names=["Non-Toxic", "Toxic"]))
print(f"Combiend F1-Score: {metrics.f1_score(val_Y, clf.predict(transformedValQuestions))}")

print("Creating Submission")
submission = test_df[['qid']].assign(prediction=pd.Series(output).values)
submission.to_csv('submission.csv', index=False)