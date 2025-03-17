import numpy as np
import json
import pickle
import nltk
from nltk.stem.snowball import FrenchStemmer
from sympy import div
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

nltk.download('punkt_tab')
stemmer = FrenchStemmer()
intents = json.loads(open('data/intents.json', encoding = 'utf-8').read())
words = pickle.load(open('data/words.pkl', 'rb'))
classes = pickle.load(open('data/classes.pkl', 'rb'))
model = load_model('models/chatbot-model-about-diabete.h5')

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)

    bag = [0] * len(words)
    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bag = bag_of_words(sentence)

    predictions = model.predict(np.array([bag]), verbose = 0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[index, result] for index, result in enumerate(predictions) if result > ERROR_THRESHOLD]
    results.sort(key = lambda x: x[1], reverse = True)

    return_list = []
    for result in results:
        return_list.append({'intent': classes[result[0]], 'probability': str(result[1])})
        return return_list

def get_response(intents_list, intents_json):
    try:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        result = ''
        for intent in list_of_intents:
            if intent['tag'] == tag:
                result = np.random.choice(intent['responses'])
                break
        return result
    except:
        return f'Je ne comprends pas..., pouvez-vous reformuler svp!'

def chatbot_response(question) :
    if not question :
        return "Désolé, je n'ai pas compris votre message."
    intents_list = predict_class(question)
    answer = get_response(intents_list, intents)
    return answer


st.markdown('<div style="text-align: center; font-size: 40px; font-weight: bold;">Chatbot Diabète</div>', unsafe_allow_html = True)
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    with st.chat_message(name = 'assistant'):
        st.markdown('Hello👋, comment puis-je vous aider?')


for message in st.session_state.messages:
    with st.chat_message(name = message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input('Comment puis-je vous aider?'):
    with st.chat_message(name = 'user'):
        st.markdown(prompt)
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })

    answer = chatbot_response(prompt)
    with st.chat_message(name = 'assistant') :
        st.markdown(answer)
    st.session_state.messages.append({
        'role' : 'assistant',
        'content' : answer
    })