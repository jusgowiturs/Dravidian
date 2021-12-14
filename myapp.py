from flask import Flask, render_template,request
#print("Code executed successfully FLASK")
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from keras.preprocessing.text import tokenizer_from_json
#import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer,text_to_word_sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
#from keras.models import load_model
#import string
from nltk.tokenize import word_tokenize

import json

app = Flask(__name__)

model = keras.models.load_model("./model-001-0.815760.h5")
f = open('Tokenizer.json')
T1 = json.load(f)
tokenizer_obj = keras.preprocessing.text.tokenizer_from_json(T1)

@app.route("/")
def hello():
	return render_template("index.html")

@app.route("/sub",methods  =["POST"])
def submit():
	max_length = 300
	if request.method == "POST":
		string = request.form["username"]
		test_sequence = tokenizer_obj.texts_to_sequences([word_tokenize(string)])
		test_pad =pad_sequences(test_sequence, maxlen=max_length, padding='post')
		print(test_pad)
		state = model.predict(test_pad)
		state = np.argmax(state)
		if state == 0:
			state = "Not Offensive Content"
		else:
			state = "Offensive Content"
		statement = [string,state]
	return render_template("sub.html",statement = statement)

if __name__ == "__main__":
	app.run(debug=True)