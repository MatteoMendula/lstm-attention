import tensorflow as tf
from keras.models import load_model
from layers import AttentionWithContext, Addition
from keras import initializers, regularizers, constraints
import tf2onnx
import onnx

onnx_model_name = './final_model/stress_lstm_plain.onnx'
model = load_model('./final_model/1_stress_lstm_plain.h5', 
                   custom_objects={
                        'AttentionWithContext':AttentionWithContext,
                        'Addition':Addition,
                        'GlorotUniform': initializers.get('glorot_uniform')
                    })

input_signature = [tf.TensorSpec([None,35], tf.int64, name='x')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, onnx_model_name)

# onnx_model = keras2onnx.convert_keras(model, model.name)
# onnx.save_model(onnx_model, onnx_model_name)