import onnxruntime as ort
import numpy as np
import util as util

import pickle


onnx_model_name = './final_model/stress_lstm_plain.onnx'
sess = ort.InferenceSession(onnx_model_name, providers=["CUDAExecutionProvider"])

word_to_idx = {}

# Read dictionary pkl file
with open("." + "/preprocessed/word_to_idx.pkl", 'rb') as fp:
    word_to_idx = pickle.load(fp)

# input
input1 = input()
print("received string: ", input1)  # print the received string

input1 = util.preprocess_lowercase_negation(input1)

words = input1.split()  # split the string into words
print("words: ", words)  # print the words

n_skipped = 0
words_embedded = []

for w in words:
    if w not in word_to_idx:
        print("word not in dictionary skipping: ", w)
        n_skipped += 1
        continue
    words_embedded.append(word_to_idx[w])

print("n_skipped: ", n_skipped)  # print the number of skipped words
print("words_embedded: ", words_embedded)  # print the embedded words

# pad with zeros
if len(words_embedded) < 35:
    words_embedded = np.pad(words_embedded, (0, 35 - len(words_embedded)), 'constant', constant_values=0)

words_embedded = np.array([words_embedded])
words_embedded = words_embedded[:35] # truncate to 35 words

print("words_embedded: ", words_embedded)  # print the embedded words
print("words_embedded shape: ", words_embedded.shape)  # print the embedded words


results_ort = sess.run(None, {"x": words_embedded})
print("results_ort: ", results_ort)  # print the results


