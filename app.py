import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from googletrans import Translator
import streamlit as st

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')

# Load and preprocess data
@st.cache_data
def load_data():
    df_lang_1 = pd.read_csv('dataset.csv')
    df_lang_2 = pd.read_csv('Language Detection.csv')
    df_lang_2 = df_lang_2.rename(columns={"Language": "language"})
    df_lang = pd.concat([df_lang_1, df_lang_2], ignore_index=True)
    return df_lang

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    text = ''.join(char for char in text if char not in punctuation and not char.isdigit())
    words = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Load data and preprocess
df_lang = load_data()
df_lang['preprocessed_text'] = df_lang['Text'].apply(preprocess_text)

# Prepare text and labels
text = df_lang['preprocessed_text']
language = df_lang['language']

# Vectorization and encoding
vectorizer = TfidfVectorizer()
label_encoder = LabelEncoder()
text_v = vectorizer.fit_transform(text)
language_v = label_encoder.fit_transform(language)

# Train the model
model = SVC()
model.fit(text_v, language_v)

# Supported languages dictionary
supported_languages = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "nl": "Dutch",
    "ar": "Arabic",
    "tr": "Turkish",
    "ta": "Tamil",
    "hi": "Hindi",
    "ro": "Romanian",
    "fa": "Persian",
    "ps": "Pushto",
    "sv": "Swedish",
    "et": "Estonian",
    "ko": "Korean",
    "zh": "Chinese",
    "pt": "Portuguese",
    "id": "Indonesian",
    "ur": "Urdu",
    "la": "Latin",
    "ja": "Japanese",
    "th": "Thai",
    "it": "Italian",
    "ml": "Malayalam",
    "de": "German",
    "da": "Danish",
    "kn": "Kannada",
    "el": "Greek"
}

# Define the Streamlit app
st.title("Language Detection and Translation")

# Text input for the user
input_text = st.text_area("Enter your text here:")

# Button for language detection
if st.button("Detect Language"):
    if input_text.strip():
        input_text_v = vectorizer.transform([preprocess_text(input_text)])
        predicted_language = label_encoder.inverse_transform(model.predict(input_text_v))
        st.write(f"The predicted language is: {predicted_language[0]}")
    else:
        st.write("Please enter some text to detect the language.")

# Language selection dropdown
target_language = st.selectbox("Select target language", list(supported_languages.keys()), format_func=lambda x: supported_languages[x])

# Button for translation
if st.button("Translate"):
    if input_text.strip():
        translator = Translator()
        try:
            translation = translator.translate(input_text, dest=target_language)
            st.write(f"Translated text: {translation.text}")
        except Exception as e:
            st.write(f"Error translating text: {e}")
    else:
        st.write("Please enter some text to translate.")
