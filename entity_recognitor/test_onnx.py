import torch
import os
import onnxruntime
import numpy as np
import time

from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings

onnx_model_path = "./flert-embeddings.onnx"
input_texts = ["I want to book this hotel"]

def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)

base_embeddings = TransformerWordEmbeddings(
            "bert-base-uncased",
            layers="-1,-2,-3,-4",
            layer_mean=False,
            allow_long_sentences=True,
            force_device=torch.device("cpu")
        )

# device = torch.device("cuda")

# get inputs
sentences = [Sentence("to speak to a customer service advisor")]
tensors = base_embeddings.prepare_tensors(sentences)
# print(tensors)


# # run onnx session
sess = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'] )
# input_name0 = sess.get_inputs()[0].name
# input_name1 = sess.get_inputs()[1].name
inputs = [x.name for x in sess.get_inputs()]
# ['input_ids', 'token_lengths', 'attention_mask', 'overflow_to_sample_mapping', 'word_ids']
# get output
outputs = [x.name for x in sess.get_outputs()]
# token_embeddings
# print(inputs)
# print(outputs)

# k: v.numpy() for k, v in tensors.items()

# start = time.time()
features = sess.run(outputs, {k: v.cpu().numpy() for k, v in tensors.items()})[0]
print(features.shape)
            # {k: dict_tensors[k].cpu().numpy() for k in inputs})['token_embeddings']
# end = time.time() - start
# print(end)
# # get ids
# probabilities = softmax(logits, axis=1)
# predicted_ids = np.argmax(probabilities, axis=1)

# # map ids to label
# predicted_labels = label_encoder.inverse_transform(predicted_ids)
# for text, pred in zip(input_texts, predicted_labels):
#     print(f"Text: {text} - Predicted: {pred}")


