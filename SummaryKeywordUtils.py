from db_connection import logger, collection
import os
import fitz  
import re 
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from string import punctuation
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
nlp = spacy.load("en_core_web_sm")

def get_pdf_size_category(file_path):
    """Categorize PDF based on number of pages"""
    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()

        if page_count <= 30:
            return 'short', 5
        elif page_count <= 50:
            return 'medium', 8
        elif page_count <= 100:
            return 'long', 12
        else:
            return 'too_long', 15

    except Exception as e:
        logger.info(f"Error checking PDF size for {file_path}: {str(e)}")
        return None

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ''
        for page in doc:
            text += page.get_text() 
        md={}
        md['pdf_size'] = f"{round(os.path.getsize(file_path)/(1024))} KB"
        md['pdf_path'] = file_path
        md['document_name'] = file_path.split('/')[-1]
        md_u = doc.metadata
        md_u.update(md)
        doc.close()

        return text, md_u

    except Exception as e:
        logger.info(f"Error processing {file_path}: {str(e)}")
        return None

def clean_text(text)-> str:
    # regex pattern for unwanted character and text 
    unwanted_pattern = r"[Ââ1]|[^\w\s]|(?<!\w)[a-hj-zA-HJ-Z](?!\w)"
    hindi_word = r"[ँ-ःअ-ऍए-ऑओ-नप-रलळव-ह़-ॅे-ॉो-्ॐ]"
    page_num = r"\d+\s+of\s+\d+" # 2 of 10
    result = ''
    clean_content = re.sub(unwanted_pattern, '', text)
    clean_content = re.sub(hindi_word, '', clean_content)
    clean_content = re.sub(r'\s+', ' ', clean_content)
    clean_content = re.sub(page_num, '', clean_content)
    result = clean_content.strip()
    text_size = len(result.split(' '))

    return result, text_size


def clean_text_for_sum(text: str) -> str:
    # Regex pattern for unwanted characters and text
    unwanted_pattern = r"[Ââ1]|[^\w\s.,!?]|(?<!\w)[a-hj-zA-HJ-Z](?!\w)"
    hindi_word = r"[ँ-ःअ-ऍए-ऑओ-नप-रलळव-ह़-ॅे-ॉो-्ॐ]"
    page_num = r"\d+\s+of\s+\d+"  # Matches patterns like "2 of 10"
    
    # Clean the text
    clean_content = re.sub(unwanted_pattern, '', text)  # Remove unwanted characters
    clean_content = re.sub(hindi_word, '', clean_content)  # Remove Hindi characters
    clean_content = re.sub(r'\s+', ' ', clean_content)  # Replace multiple spaces with a single space
    clean_content = re.sub(page_num, '', clean_content)  # Remove page number patterns
    result = clean_content.strip()  # Strip leading and trailing whitespace

    return result


def words_strength(text):
    doc = nlp(text.lower())
    keyword = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if(token.text in stopwords or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keyword.append(token.text)
    return Counter(keyword)

def sentence_strength(text):
    sent_strength={}
    doc = nlp(text.lower())
    freq_word = words_strength(text=text)
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]
    return sent_strength


def summarize(text, n=4):
    try:
        summary = []
        sent_strength = sentence_strength(text)  # this returns a dict of sentences and their strengths
        sorted_sentences = sorted(sent_strength.items(), key=lambda x: x[1], reverse=True)
        le = len(sorted_sentences)
        sum_size = int(le*0.2)
        n = min(sum_size, 500) # capping the summary upto 500 sentences 
        # Collecting the top n sentences, converting them to text if they are not already strings
        for i in range(min(n, len(sorted_sentences))):  # Ensuring we don't exceed the number of sentences
            sentence = sorted_sentences[i][0]
            # Convert to string if it's a spacy token or similar object
            if hasattr(sentence, 'text'):
                summary.append(sentence.text)
            else:
                summary.append(str(sentence))  # Fallback to str conversion

        logger.info("Summarization is done from scratch, not with LLM")
        return " ".join(summary)
    except Exception as e:
        logger.error(f'Error summarizing text: {str(e)}')
        return []

def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN'] 
    doc = nlp(text.lower()) 
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result


def add_metadata(metadata):
    doc = {
        'filename': metadata['document_name'],
        'metadata': metadata
    }
    try:
        collection.insert_one(doc)
        logger.info('Added metadata to the database')
    except Exception as e:
        logger.error(f'Error Pushing to database: {str(e)}')

def update_db(file_name, update_data):
    try:
        collection.update_one(
                {"filename": file_name},
                update_data
            )
        logger.info('Updated database for summary and keywords')
    except Exception as e:
        logger.error(f'Error updating database: {str(e)} for summary and keywords {file_name}')
        raise


def process_single_pdf(file_path):
    """Process a single PDF with appropriate number of workers"""
    try:
        _, num_workers = get_pdf_size_category(file_path)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future = executor.submit(extract_text_from_pdf, file_path)
            # This will return the pdf text and pdf metadata (text, metadata)
            pdf_text, pdf_metadata = future.result()
            add_metadata(pdf_metadata)
            logger.info(f"Text is extracted from {pdf_metadata['document_name']}.")

        # Clean the text for summerisation and keyword extraction
        cleaned_text, text_size = clean_text(pdf_text)
        cleaned_text_sum = clean_text_for_sum(pdf_text)
        logger.info(f"Text cleaning is done of {pdf_metadata['document_name']}.")

        # summerize the text
        logger.info(f"Summerizing {pdf_metadata['document_name']}.")
        summary = summarize(cleaned_text_sum)
        logger.info(f"Summerization of {pdf_metadata['document_name']} is done.")

        #keyword extraction
        logger.info(f"Extracting Keywords from {pdf_metadata['document_name']}.")
        output = get_hotwords(cleaned_text)
        most_common_list = Counter(output).most_common(30)
        keywords = [key for key,_ in most_common_list]
        logger.info(f"Extracting Keywords from {pdf_metadata['document_name']} is completed.")

        update_data = {
            "$set": {
                "summary": summary,
                "keywords": keywords
            }
        }

        update_db(
            pdf_metadata['document_name'],
            update_data
        )
        logger.info("Updated the database with summary and keywords")
    
    except Exception as e:
        logger.error(f"Some unknown error happened in the pipeline {e}")
        raise e

