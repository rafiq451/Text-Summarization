import streamlit as st
import numpy as np

from model import fetch_article, casefolding, cleaning, tokenization, stopword_removal, sentence_split, word_freq, sentence_weight, detect_language

st.title("Text Summarization App")
st.write("Masukkan URL artikel untuk dirangkum.")

# Input URL
url = st.text_input("Masukkan URL artikel:")
n = st.slider("Pilih jumlah kalimat ringkasan:", min_value=1, max_value=10, value=2)

if st.button("Ringkas Artikel"):
    if url:
        news = fetch_article(url)
        if news.startswith("Error"):
            st.error(news)
        else:
            st.write("**Artikel Asli:**")
            st.write(news)

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
                st.warning("Tidak ada data yang dapat diproses setelah preprocessing.")
            else:
                wordfreq = word_freq(data)
                rank = sentence_weight(data, wordfreq)

                # Ringkasan
                result = ''
                sort_list = np.argsort(rank)[::-1][:n]
                for i in sort_list:
                    result += '{} '.format(sentence_list[i])

                st.subheader("Ringkasan Artikel:")
                st.write(result)
    else:
        st.warning("Masukkan URL terlebih dahulu.")
