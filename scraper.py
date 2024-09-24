import re
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from tabulate import tabulate
import matplotlib.pyplot as plt

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

slang_dict = {
    "cocaine": ["snow", "blow", "coke", "bump"],
    "marijuana": [
        "weed", "ganja", "kush", "bud", "420", "vape", 
        "joint", "blunt", "mary jane", "purple drank"
    ],
    "ecstasy": ["molly", "mdma", "e", "roll"],
    "heroin": ["dope", "smack"],
    "hashish": ["hash"],
    "LSD": ["acid", "tabs", "shrooms", "trip"],
    "amphetamine": ["speed"],
    "methamphetamine": ["meth", "crystal"],
    "alprazolam": ["xanax", "bars"],
    "benzodiazepines": ["benzos"],
    "codeine": ["lean"],
    "oxycodone": ["oxy"],
    "percocet": ["perc"],
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    for substance, slangs in slang_dict.items():
        for slang in slangs:
            text = text.replace(slang, substance)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)

def process_and_predict(scraped_data):
    df = pd.DataFrame(scraped_data)
    df['cleaned_text'] = df['text'].apply(clean_text)
    X_tfidf = vectorizer.transform(df['cleaned_text'])
    predictions = model.predict(X_tfidf)
    df['prediction'] = predictions
    return df

chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

url = 'http://127.0.0.1:3000/chat.html'
driver.get(url)

driver.implicitly_wait(10)
html_content = driver.page_source
driver.quit()

soup = BeautifulSoup(html_content, 'html.parser')

def scrape_chat_data():
    chat_data = []
    messages = soup.select('#chat .message')
    print(f"\nNumber of messages found: {len(messages)}")
    
    for message in messages:
        username = message.select_one('.username')
        text = message.select_one('.text')
        
        if username and text:
            username = username.get_text(strip=True).replace(':', '')
            text = text.get_text(strip=True)
            chat_data.append({'username': username, 'text': text})
        else:
            print("Skipping a message due to missing username or text.")

    return chat_data

scraped_data = scrape_chat_data()
print("\nScraped Data:")
print(scraped_data)


if scraped_data:
    result_df = process_and_predict(scraped_data)
    
    print("\nProcessed and Predicted Data:")
    print(tabulate(result_df[['username', 'text', 'prediction']], headers='keys', tablefmt='pretty'))

    for index, row in result_df.iterrows():
        if row['prediction'] == 1:
            print(f"Message to store: '{row['text']}', Sender: '{row['username']}'")
            
    result_df.to_csv('processed_chat_data.csv', index=False)
    print("Processed data exported to 'processed_chat_data.csv'.")
       
    prediction_counts = result_df['prediction'].value_counts()    
    plt.bar(prediction_counts.index, prediction_counts.values, color=['red', 'green'])
    plt.title('Message Predictions')
    plt.xlabel('Prediction (0 = Non-flagged, 1 = Flagged)')
    plt.ylabel('Number of Messages')
    plt.xticks([0, 1], ['Non-flagged', 'Flagged'])
    plt.show()
else:
    print("No valid chat data to process.")


