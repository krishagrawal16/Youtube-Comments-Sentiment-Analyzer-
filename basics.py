import re
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import emoji
from flask import Flask, render_template, request
from googleapiclient.discovery import build

# Define all the functions used in the pipeline
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

exclude = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
def remove_punc1(text):
    if not isinstance(text, str):
        text = str(text)
    return text.translate(str.maketrans('', '', ''.join(exclude)))  # Convert set to string

chat_words = {
    'ive':'i have', 'couldnt':'could not', 'id':'i had', 'im':'i am', 'dont':'do not',
    'its':'it is', 'ur':'your', 'asap':'as soon as possible', 'isnt':'is not', 'thx':'thanks',
    'ig':'i guess', 'nyc':'New York', 'm':'am', 'omg':'oh my god', 'u':'you'
}

def chat_conversion(text):
    new_text = []
    for w in text.split():
        if w.lower() in chat_words:
            new_text.append(chat_words[w.lower()])
        else:
            new_text.append(w)
    return " ".join(new_text)

lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text.split()])


def fillna_transformer(x):
    return x.fillna('').astype(str)

def lowercase_transformer(x):
    return x.apply(lambda x: x.lower())

def remove_url_transformer(x):
    return x.apply(remove_url)

def remove_html_tags_transformer(x):
    return x.apply(remove_html_tags)

def remove_punc_transformer(x):
    return x.apply(remove_punc1)

def chat_conversion_transformer(x):
    return x.apply(chat_conversion)

def lemmatize_text_transformer(x):
    return x.apply(lemmatize_text)

# Load the pipeline and model
text_cleaning_pipeline = pickle.load(open('text_cleaning_pipeline.pkl', 'rb'))
model = pickle.load(open('model_Final.pkl', 'rb'))
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)

API_KEY = 'AIzaSyAwAf3pnhA9y2oi_RyYfzTnwEUZBbYBBTw'
youtube = build('youtube', 'v3', developerKey=API_KEY)

def fetch_youtube_comments(url):
    match = re.search(r'v=([^&]+)', url)
    if not match:
        return "Invalid YouTube URL"
    
    video_id = match.group(1)
    video_response = youtube.videos().list(part='snippet', id=video_id).execute()
    video_snippet = video_response['items'][0]['snippet']
    uploader_channel_id = video_snippet['channelId']
    
    comments = []
    nextPageToken = None
    while len(comments) < 37:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100, 
            pageToken=nextPageToken
        )
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            if comment['authorChannelId']['value'] != uploader_channel_id:
                comments.append(comment['textDisplay'])
        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break
    
    threshold_ratio = 0.65
    relevant_comments = []
    for comment_text in comments:
        comment_text = comment_text.lower().strip()
        emojis_count = emoji.emoji_count(comment_text)
        text_characters = len(re.sub(r'\s', '', comment_text))
        if emojis_count == 0 or (text_characters / (text_characters + emojis_count)) > threshold_ratio:
            relevant_comments.append(comment_text)
    
    relevant_comments_df = pd.DataFrame(relevant_comments, columns=['Relevant Comment'])
    relevant_comments_df.to_csv("relevant_ytcomments.csv", index=False, encoding='utf-8')

    relevant_comments_df['Relevant Comment'] = text_cleaning_pipeline.fit_transform(relevant_comments_df['Relevant Comment'])
    sequences = tokenizer.texts_to_sequences(relevant_comments_df['Relevant Comment'])
    padded_sequences = pad_sequences(sequences, maxlen=100,padding = 'post')
    predictions = model.predict(padded_sequences)

    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    result = [sentiment_labels[np.argmax(pred)] for pred in predictions]
    counts = {label: result.count(label) for label in sentiment_labels}   
    # Combine comments with their predicted sentiments
    comments_with_sentiments = list(zip(relevant_comments, result))
    
    return comments_with_sentiments, counts

@app.route('/')
def home():
    return render_template('site.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form['url']
    comments_with_sentiments, counts = fetch_youtube_comments(url)
    
    result_string = f"Analyzing comments for the URL: {url}.\n\n"
    result_string += f"Sentiment Counts:\n{counts}\n\n"
    result_string += "Comments and their sentiments:\n"
    
    for comment, sentiment in comments_with_sentiments:
        result_string += f"Comment: {comment} -> Sentiment: {sentiment}\n"
    
    return result_string

if __name__ == "__main__":
    app.run(debug=True)
