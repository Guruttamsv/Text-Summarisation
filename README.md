# Text Summarization

This project demonstrates text summarization techniques using deep learning models and natural language processing (NLP) tools. The project utilizes the Mistral-7B-Instruct-v0.3 model for text generation and summarization, focusing on summarizing speeches and documents into concise summaries.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Acknowledgements](#acknowledgements)

## Project Overview

The project aims to automatically generate summaries from large texts or speeches using advanced NLP techniques. It employs models like Mistral-7B-Instruct-v0.3 for generating summaries based on provided prompts and inputs.

## Features

* **PDF Text Extraction:** Extracts text from PDF documents for summarization.
* **Text Preprocessing:** Cleans and preprocesses text data for better summarization results.
* **Speech Summarization:** Summarizes speeches or long documents into concise summaries.
* **Interactive Prompts:** Uses prompts to guide the summarization process for specific outputs.
* **Visualization:** Visualizes the summarization results and provides insights into the generated summaries.


## System Requirements

+ **Python:** 3.6 or higher
+ **Libraries:** 
  + **langchain:** 0.1.0 or higher
  + **transformers:**  4.x
  + **PyPDF2:** 1.26.0 or higher
  + **matplotlib:** 3.3 or higher
  + **yfinance:** 0.1.63 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Guruttamsv/Text-Summarization.git
cd Text-Summarization
```
2. Set up a virtual environment (optional but recommended):
```bash
conda create -n text-summarization python=3.8
conda activate text-summarization
```
3. Install required packages:
```bash
pip install langchain transformers PyPDF2 matplotlib yfinance
```


## Usage

### 1. Data Collection and Preprocessing:
**Extracting text from PDF documents:**
```python
from PyPDF2 import PdfReader

pdfreader = PdfReader('apjspeech.pdf')
text = ''
for page in pdfreader.pages:
    content = page.extract_text()
    if content:
        text += content
```

### 2. Text Summarization Techniques:
**Using advanced NLP models for summarization:**
```python
from langchain.chains import load_summarize_chain

# Load summarization chain using Mistral-7B-Instruct-v0.3 model
chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)

# Run summarization on text chunks
summary = chain.run(chunks)
print(summary)
```

### 3. Interactive Prompts for Summarization:
**Providing specific prompts for generating summaries:**
```python
from langchain import PromptTemplate

map_prompt_template = PromptTemplate(input_variables=['text'], template="""
Please summarize the below speech:
Speech:`{text}'
Summary:
""")
final_combine_prompt_template = PromptTemplate(input_variables=['text'], template="""
Provide a final summary of the entire speech with these important points.
Add a Generic Motivational Title,
Start the precise summary with an introduction and provide the
summary in number points for the speech.
Speech: `{text}`
""")
```

## Results

* **Speech Summarization:** Generates concise summaries from long speeches or documents.
* **PDF Text Extraction:** Extracts text from PDFs for further analysis and summarization.
* **Interactive Prompts:** Guides the summarization process with specific input prompts for tailored outputs.

## Limitations and Future Work

### Limitations
* **Model Accuracy:** The accuracy of the summarization heavily depends on the model's training data and architecture.
* **Data Quality** Large datasets with diverse texts may improve the model's performance.

### Future Work
* **Enhanced Models:** Explore more advanced NLP models like GPT-3 or BERT for better summarization accuracy.
* **Real-time Summarization:** Implement real-time summarization for live speeches or news articles.
* **Multilingual Support:** Extend the model to support summarization in multiple languages for broader applications.

## Acknowledgements

* **Mistral-7B-Instruct-v0.3:** For providing the deep learning model used in this project.
* **PyPDF2:** For PDF text extraction capabilities.
* **Google Colab:** For providing a collaborative platform for running and testing the code.
* **Statsmodels:** For providing ARIMA modeling functionality.

