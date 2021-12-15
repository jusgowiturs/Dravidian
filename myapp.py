from flask import Flask, render_template,request
#print("Code executed successfully FLASK")
import numpy as np
import tensorflow as tf
#from tensorflow import keras
#from keras.preprocessing.text import tokenizer_from_json
#import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer,text_to_word_sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
#from keras.models import load_model
#import string
#from nltk.tokenize import word_tokenize

import json

app = Flask(__name__)

#model = tf.keras.models.load_model("./model-001-0.815760.h5")
model = tf.keras.models.load_model("./model-001-0.617347.h5")
f = open('Tokenizer.json')
T1 = json.load(f)
tokenizer_obj = tf.keras.preprocessing.text.tokenizer_from_json(T1)

@app.route("/")
def hello():
	return render_template("index.html")

@app.route("/sub",methods  =["POST"])
def submit():
	max_length = 300
	if request.method == "POST":
		string = request.form["username"]
		#test_sequence = tokenizer_obj.texts_to_sequences([word_tokenize(string)])
		test_sequence = tokenizer_obj.texts_to_sequences(string.split(' '))
		#test_sequence = sum(test_sequence,[])
		print(test_sequence)
		test_pad =pad_sequences(test_sequence, maxlen=max_length, padding='post')
		print(test_pad)
		state = model.predict(test_pad)
		state = np.argmax(state,axis=1)
		NO = np.count_nonzero(state == 0)
		O = np.count_nonzero(state == 1)
		print("STATE",NO,O)
		if NO > O:
			state = "Not Offensive Content"
		if O > NO:
			state = "Offensive Content"
		if NO==O:
			state = "In Distinguishable"
			
		statement = [string,state]
	return render_template("sub.html",statement = statement)

if __name__ == "__main__":
	app.run(debug=True)