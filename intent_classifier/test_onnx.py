import torch
import os
import onnxruntime
import numpy as np
import pickle
from transformers import RobertaTokenizer, RobertaForSequenceClassification

label_path = "./label_encoder/label_encoder.pkl"
onnx_model_path = "./checkpoints/model.onnx"
input_texts = ["I want to book this hotel"]

def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)


device = torch.device("cuda")

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

with open(label_path, 'rb') as f:
    label_encoder = pickle.load(f)

dummy_model_input = tokenizer(input_texts, padding='max_length', truncation=True, return_tensors="pt")

# get inputs
input_ids = dummy_model_input['input_ids']
attention_mask = dummy_model_input['attention_mask']

# print(input_ids.shape)
# print(attention_mask.shape)
# print(onnxruntime.get_device())

# run onnx session
sess = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'] )
input_name0 = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name

# get output
outputs = [x.name for x in sess.get_outputs()]

logits = sess.run(outputs, 
            {input_name0: np.array(input_ids),
            input_name1: np.array(attention_mask)})[0]

# get ids
probabilities = softmax(logits, axis=1)
predicted_ids = np.argmax(probabilities, axis=1)

# map ids to label
predicted_labels = label_encoder.inverse_transform(predicted_ids)
for text, pred in zip(input_texts, predicted_labels):
    print(f"Text: {text} - Predicted: {pred}")


