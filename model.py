import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

data=pd.read_csv('chat_data.csv')
df = pd.DataFrame(data)

slang_dict = {
    "marijuana": ["weed", "ganja", "grass", "mary jane", "420", "kush", "bud", "vape", "joint", "blunt", "maal"],
    "cocaine": ["snow", "blow", "coke", "bump"],
    "ecstasy": ["molly", "mdma", "e", "roll"],
    "heroin": ["dope", "smack"],
    "methamphetamine": ["meth", "crystal"],
    "lsd": ["acid", "tabs", "trip"],
    "oxycodone": ["oxy"],
    "percocet": ["perc"],
    "alprazolam": ["xanax", "bars"],
    "benzodiazepines": ["benzos"],
    "codeine": ["lean", "purple drank"]
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)

    for substance, slang_list in slang_dict.items():
        for slang in slang_list:
            text = text.replace(slang, substance)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# y_pred = model.predict(X_test_tfidf)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.2f}")
