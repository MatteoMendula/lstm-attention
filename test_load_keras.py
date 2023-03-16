import keras
import numpy as np

from layers import AttentionWithContext, Addition
from keras import initializers


model_loaded = keras.models.load_model('./final_model/1_stress_lstm_plain.h5', 
                   custom_objects={
                        'AttentionWithContext':AttentionWithContext,
                        'Addition':Addition,
                        'GlorotUniform': initializers.get('glorot_uniform')
                    })

interview_test_data = np.load("." + "/preprocessed/interview_test_data.npy")
print(interview_test_data.shape)
print(interview_test_data[0])

index_to_try = 111

single_test = np.array([interview_test_data[index_to_try]])

test_predictions = model_loaded.predict(single_test, verbose=0)

print(test_predictions)  # print the results

test_predictions = model_loaded.predict(interview_test_data, verbose=0)
print(test_predictions[index_to_try])  # print the results