# Lang-d-t: Language Detection and Translation

A Gradio web app that detects the language of any text you enter and translates it into a language you choose.

**Stack:** Python · Gradio · scikit-learn · NLTK · pandas · googletrans

## Problem

When you come across text in a language you can't identify, you have to guess the language before you can translate it. This app does both steps for you: it identifies the language first, then translates.

## Features

- **Language detection:** identifies the language of the input text using a machine learning model trained with scikit-learn
- **Translation:** translates the text into a target language of your choice
- **Simple interface:** built with Gradio, so it runs in the browser
- **Multiple languages:** supports a wide range of languages for detection and translation

## How it works

1. The input text is cleaned and preprocessed with NLTK.
2. The trained scikit-learn model predicts the language of the text.
3. The text is translated into the selected target language using the `googletrans` library.
4. The detected language and the translation are shown in the interface.

## Tech stack

| Purpose | Technology |
|---|---|
| Language | Python |
| Web interface | Gradio |
| Language detection model | scikit-learn |
| Text preprocessing | NLTK |
| Data handling | pandas |
| Translation | googletrans |

## Limitations

- Very short or mixed-language text is harder to detect correctly.
- `googletrans` is an unofficial wrapper around Google Translate, so it can break or be rate-limited without notice.

## Next steps

- Show a confidence score with each detection
- Switch to an official translation API for reliability
- Handle mixed-language input

## Author

Dharini M ([@dharini03](https://github.com/dharini03))
