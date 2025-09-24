import warnings
import pandas as pd
import nltk
import re
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords


print("Hello Github!")



def lemmatize(inp):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(inp)])

def punctuationRemoval(text):
    return text.translate(str.maketrans("","", punctuation))

def stopwordsRemoval(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])


warnings.filterwarnings('ignore')


#Read the data
df = pd.read_csv("Tweets.csv")
df = df.dropna(subset=["airline_sentiment"])
df = df[["airline_sentiment", "airline_sentiment_confidence", "text"]]
df = df[df["airline_sentiment"] != "neutral"]

#Oversampling
df_positive = df[df["airline_sentiment"] == "positive"]
df_negative = df[df["airline_sentiment"] == "negative"]

# Oversample the minority class (positive)
df_positive_oversampled = df_positive.sample(n=len(df_negative), replace=True, random_state=42)

# Combine both
df_balanced = pd.concat([df_negative, df_positive_oversampled])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

df = df_balanced
print(df["airline_sentiment"].value_counts())
# Preprocessing the data
print()
print(df["text"][0])
print()

df["text"] = df["text"].apply(lambda x: " ".join(x.split()[1:])) #remove the username
df["text"] = df["text"].str.lower() # lowering all the cases
df["text"] = df["text"].apply(lambda x: punctuationRemoval(x))# remove any punctuations
df["text"] = df["text"].apply(lambda x: stopwordsRemoval(x))
df["text"] = df["text"].apply(lambda x: lemmatize(x)) # Lemmatization

print(df["text"][0])
print()


#One Hot Encoding (Mapping)
sentiment = {"positive": 1,
            "negative": 2
            }

y = df["airline_sentiment"].map(sentiment)

# Split the dataset (30%, 70%)
X_train, X_test, y_train, y_test = train_test_split(df["text"], y, test_size = 0.3, random_state = 43 )

# Word embedding using bag of wordS(BoW), TF-IDF
tf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=5, stop_words='english')
tf_X_train = tf_vectorizer.fit_transform(X_train) #TF-IDF, BoW vectorizers expect a raw data as input, whereas Word2vec require tokenized input

bow_vectorizer = CountVectorizer()
bow_X_train = bow_vectorizer.fit_transform(X_train) #TF-IDF, BoW vectorizers expect a raw data as input, whereas Word2vec require tokenized input (if no pipeline used, use this as training x )


#Models Building


clf_rl = LogisticRegression(class_weight='balanced',C=1.0, penalty='l2', solver='liblinear')
'''
C	Inverse of regularization strength. Lower = stronger regularization. Try: [0.01, 0.1, 1, 10, 100]
penalty	Regularization type: 'l2' is default and often best for text. 'l1' can give sparse models.
solver	'liblinear' works well for small datasets and allows 'l1'. 'saga' supports both 'l1' and 'l2', better for large datasets.
max_iter	Increase if model doesn't converge. Try: 1000, 2000, etc.
'''

clf_svc = LinearSVC(class_weight='balanced', C=1.0, penalty='l2', loss='squared_hinge', dual=True)
'''
C	Regularization (same as LogisticRegression). Try: [0.01, 0.1, 1, 10]
loss	Use 'squared_hinge' (default) or 'hinge' (rarely better)
penalty	Always 'l2' for LinearSVC
dual	For large feature sets (TF-IDF) and few samples, use True. For many samples, False may be faster.
max_iter	Increase to avoid convergence warnings. Try 1000, 2000, etc.
'''

clf_cnb = ComplementNB(alpha=1.0, fit_prior=True, norm=False)

'''
alpha	Same as in MultinomialNB — smoothing. Often works well with small values like 0.1
fit_prior	Same as above
norm	Normalize weights — can improve performance when using TF-IDF. Try both True and False.

🔎 ComplementNB is often
better than MultinomialNB for imbalanced datasets.
'''
clf_mnb = MultinomialNB(alpha=1.0, fit_prior=True)
'''
alpha	Smoothing — prevents zero probabilities. Lower values (e.g., 0.1, 0.01) can improve accuracy. Try: [1.0, 0.5, 0.1, 0.01]
fit_prior	Whether to learn class prior probabilities from training data. Sometimes setting False improves results when classes are balanced.

'''

#Model1: TF-IDF & LinearSVC
pipeline_svc_tfidf = Pipeline([
    ("tfidf", tf_vectorizer),
    ("clf", clf_svc)
        ])

#Model2: BoW & LinearSVC
pipeline_svc_bow = Pipeline([
    ("bow", bow_vectorizer),
    ("clf", clf_svc)
        ])

#Model3: TF-IDF & Compliment NB
pipeline_cnb_tfidf = Pipeline([
    ("tfidf", tf_vectorizer),
    ("clf", clf_cnb)
        ])

#Model4: BoW & Complement NB
pipeline_cnb_bow = Pipeline([
    ("bow", bow_vectorizer),
    ("clf", clf_cnb)
        ])

#Model5: TF-IDF & Myltinomial NB
pipeline_mnb_tfidf = Pipeline([
    ("tfidf", tf_vectorizer),
    ("clf", clf_mnb)
        ])

#Model6: BoW & Myltinomial NB
pipeline_mnb_bow = Pipeline([
    ("bow", bow_vectorizer),
    ("clf", clf_mnb)
        ])

#Model7: TF-IDF & Logistic Regression
pipeline_rl_tfidf = Pipeline([
    ("tfidf", tf_vectorizer),
    ("clf", clf_rl)
        ])

#Model8: BoW & Logistic Regression
pipeline_rl_bow = Pipeline([
    ("bow", bow_vectorizer),
    ("clf", clf_rl)
        ])




# Training and Testing all the models

pipeline_svc_tfidf.fit(X_train, y_train)
y_pred_model1 = pipeline_svc_tfidf.predict(X_test)


pipeline_svc_bow.fit(X_train, y_train)
y_pred_model2 = pipeline_svc_bow.predict(X_test)

pipeline_cnb_tfidf.fit(X_train, y_train)
y_pred_model3 = pipeline_cnb_tfidf.predict(X_test)

pipeline_cnb_bow.fit(X_train, y_train)
y_pred_model4 = pipeline_cnb_bow.predict(X_test)

pipeline_mnb_tfidf.fit(X_train, y_train)
y_pred_model5 = pipeline_mnb_tfidf.predict(X_test)

pipeline_mnb_bow.fit(X_train, y_train)
y_pred_model6 = pipeline_mnb_bow.predict(X_test)

pipeline_rl_tfidf.fit(X_train, y_train)
y_pred_model7 = pipeline_rl_tfidf.predict(X_test)

pipeline_rl_bow.fit(X_train, y_train)
y_pred_model8 = pipeline_rl_bow.predict(X_test)


# Combact the findings

Vectorizor_used = ["TF-IDF",
                   "BoW",
                   "TF-IDF",
                   "BoW",
                   "TF-IDF",
                   "BoW",
                   "TF-IDF",
                   "BoW"
                  ]
algorithem_used = ["LinearSVC",
                   "LinearSVC",
                   "ComplimentNB",
                   "ComplimentNB",
                   "MultinomialNB",
                   "MultinomialNB",
                   "Logistic Regression",
                   "Logistic Regression"
                    ]
Acc = [accuracy_score(y_test, y_pred_model1),
       accuracy_score(y_test, y_pred_model2),
       accuracy_score(y_test, y_pred_model3),
       accuracy_score(y_test, y_pred_model4),
       accuracy_score(y_test, y_pred_model5),
       accuracy_score(y_test, y_pred_model6),
       accuracy_score(y_test, y_pred_model7),
       accuracy_score(y_test, y_pred_model8)
      ]


Acc = [f'{100 * x:.2f}%' for x in Acc]
f = {"Vectorizer": Vectorizor_used,
     "Algorithem": algorithem_used,
     "Accuracy" : Acc}

findings = pd.DataFrame(f)


#Shows the comparission
print(findings)