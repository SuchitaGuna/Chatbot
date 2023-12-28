import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import scrolledtext

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('C:\python projects\chatbot\intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Zeena The Chatbot")

        # Increase the size of the window
        master.geometry("600x400")

        # Change the color scheme
        master.configure(bg="#F5F5F5")

        self.chatbox = scrolledtext.ScrolledText(
            master, wrap=tk.WORD, width=50, height=15, font=("Arial", 12)
        )
        self.chatbox.pack(padx=20, pady=20)

        self.input_entry = tk.Entry(master, width=40, font=("Arial", 12))
        self.input_entry.pack(pady=10)

        self.send_button = tk.Button(
            master, text="Send", command=self.send_message, bg="#4CAF50", fg="white"
        )
        self.send_button.pack(pady=10)

    def send_message(self):
        user_message = self.input_entry.get()
        self.input_entry.delete(0, tk.END)
        self.display_message(f"User: {user_message}")

        # Get chatbot response
        intents_list = predict_class(user_message)
        chatbot_response = get_response(intents_list, intents)
        self.display_message(f"Chatbot: {chatbot_response}")

    def display_message(self, message):
        current_text = self.chatbox.get("1.0", tk.END)
        self.chatbox.delete("1.0", tk.END)
        self.chatbox.insert(tk.END, current_text + message + "\n")
        self.chatbox.yview(tk.END)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

if __name__ == "__main__":
    root = tk.Tk()
    chatbot_gui = ChatbotGUI(root)
    print("GO! Bot is running!")
    root.mainloop()
