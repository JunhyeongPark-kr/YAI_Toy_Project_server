from django.conf import settings
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from .dataset import papers_dataset
import nltk

def load_model():  
    model_path = settings.MODEL_PATH
    model = SentenceTransformer(model_path)
    return model

model = load_model()

def create_sentence_dataset(papers_dataset):
    sentence_dataset = []

    for section in papers_dataset['body_text']:
        if 'text' in section:
            section_text = " ".join(section['text'])
            # 문장 단위로 분할하고 결과 리스트에 추가
            sentences = sent_tokenize(section_text)
            sentence_dataset.extend(sentences)

    return sentence_dataset

sentence_dataset = create_sentence_dataset(papers_dataset)


def predict(input_text):
    sentence_embeddings = model.encode(sentence_dataset, convert_to_tensor=True)

    dimension = sentence_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(sentence_embeddings.cpu()))

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence_dataset)

    input_sentences = sent_tokenize(input_text)
    results = []  # 결과를 저장할 리스트

    for input_sentence in input_sentences:
        input_sentence_vector = model.encode([input_sentence], convert_to_tensor=True)
        D, I = index.search(np.array(input_sentence_vector.cpu()), 1)
        cosine_similarities = util.pytorch_cos_sim(input_sentence_vector, sentence_embeddings)
        input_tfidf_vector = tfidf_vectorizer.transform([input_sentence])
        tfidf_cosine_similarities = cosine_similarity(input_tfidf_vector, tfidf_matrix).flatten()

        for i, idx in enumerate(I[0]):
            results.append({
                'input_sentence': input_sentence,
                'similar_sentence': sentence_dataset[idx],
                'distance': D[0][i],
                'cosine_similarity': cosine_similarities[0][idx].item(),
                'tfidf_similarity': tfidf_cosine_similarities[idx]
            })

    return results