
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from googletrans import Translator


# Importing datasets
df_lang_1 = pd.read_csv('dataset.csv')
df_lang_2 = pd.read_csv('Language Detection.csv')
df_lang_2['Language'] = df_lang_2['Language'].replace(['Portugeese', 'Sweedish'], ['Portuguese', 'Swedish'])
df_lang_2 = df_lang_2.rename(columns={"Language": "language"})
df_lang = pd.concat([df_lang_1, df_lang_2], ignore_index=True)

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    text = ''.join(char for char in text if char not in punctuation and not char.isdigit())
    words = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Apply preprocessing
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


def language_detector(input_text):
    input_text_v = vectorizer.transform([preprocess_text(input_text)])
    return label_encoder.inverse_transform(model.predict(input_text_v))[0]

def translate_text(input_text, target_language):
    translator = Translator()
    try:
        translation = translator.translate(input_text, dest=target_language)
        return translation.text
    except Exception as e:
        return str(e)

# Gradio interface functions
def detect_language(input_text):
    return language_detector(input_text)

def perform_translation(input_text, target_language):
    return translate_text(input_text, target_language)

# Create Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Language Detection and Translation")
    
    input_text = gr.Textbox(label="Input Text", placeholder="Enter text or paragraph here...")
    
    detect_button = gr.Button("Detect Language")
    detected_language_output = gr.Textbox(label="Detected Language", interactive=False)
    
    translate_button = gr.Button("Translate")
    target_language_dropdown = gr.Dropdown(choices=list(supported_languages.values()), label="Select Target Language", value="English")
    translated_text_output = gr.Textbox(label="Translated Text", interactive=False)
    
    # Define button actions
    detect_button.click(fn=detect_language, inputs=input_text, outputs=detected_language_output)
    translate_button.click(fn=perform_translation, inputs=[input_text, target_language_dropdown], outputs=translated_text_output)

# Launch the Gradio app
iface.launch()

