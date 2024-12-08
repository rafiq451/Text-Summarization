import re
import numpy as np
from nltk.corpus import stopwords
import nltk
from bs4 import BeautifulSoup
import requests

# Download NLTK resources (harus dijalankan sekali saja)
nltk.download('stopwords')
nltk.download('punkt')

# Stopwords untuk Bahasa Inggris dan Indonesia
stopwords_eng = set(stopwords.words('english'))
stopwords_id = set(stopwords.words('indonesian'))

# Fungsi-fungsi utama
def casefolding(sentence):
    return sentence.lower()

def cleaning(sentence):
    return re.sub(r'[^a-z]', ' ', re.sub("’", '', sentence))

def tokenization(sentence):
    return sentence.split()

def stopword_removal(token, language='english'):
    stopwords = stopwords_eng if language == 'english' else stopwords_id
    return [word for word in token if word not in stopwords]

def sentence_split(paragraph):
    return nltk.sent_tokenize(paragraph)

def word_freq(data):
    w = []
    for sentence in data:
        for words in sentence:
            w.append(words)
    bag = list(set(w))
    res = {}
    for word in bag:
        res[word] = w.count(word)
    return res

def sentence_weight(data, wordfreq):
    weights = []
    for words in data:
        temp = 0
        for word in words:
            temp += wordfreq.get(word, 0)
        weights.append(temp)
    return weights

# mengambil data artikel dari url
def fetch_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        paragraphs = soup.find_all('p')
        article = ' '.join([para.get_text() for para in paragraphs])
        return article
    except Exception as e:
        return f"Error fetching article: {e}"

# Deteksi bahasa
def detect_language(text):
    english_tokens = stopword_removal(tokenization(text), 'english')
    indonesian_tokens = stopword_removal(tokenization(text), 'indonesian')
    return 'indonesian' if len(indonesian_tokens) > len(english_tokens) else 'english'


# Fungsi untuk membuat ringkasan
def summarize_article(url, n=2):
    news = fetch_article(url)
    
    if news.startswith("Error"):
        return news

    # Deteksi bahasa
    language = detect_language(news)

    # Proses teks
    sentence_list = sentence_split(news)
    data = []
    for sentence in sentence_list:
        tokens = tokenization(cleaning(casefolding(sentence)))
        data.append(stopword_removal(tokens, language))
    data = list(filter(None, data))

    if not data:
        return "Tidak ada data yang dapat diproses setelah preprocessing."

    # Hitung frekuensi kata
    wordfreq = word_freq(data)

    # Hitung bobot kalimat
    rank = sentence_weight(data, wordfreq)

    # Pilih n kalimat teratas
    result = ''
    sort_list = np.argsort(rank)[::-1][:n]
    for i in sort_list:
        result += '{} '.format(sentence_list[i])

    return result
