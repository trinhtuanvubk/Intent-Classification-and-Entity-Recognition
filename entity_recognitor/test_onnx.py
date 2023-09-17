import torch
import os
import onnxruntime
import numpy as np
import time

from flair.data import Sentence, Dictionary, Span, get_spans_from_bio
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from typing import List, Tuple, Optional, Union, Any, Dict, cast
from viterbi import ViterbiDecoder
from flair.training_utils import store_embeddings

def _determine_if_span_prediction_problem(self, dictionary: Dictionary) -> bool:
    for item in dictionary.get_items():
        if item.startswith("B-") or item.startswith("S-") or item.startswith("I-"):
            return True
    return False


# DT = typing.TypeVar("DT", bound=DataPoint)
# DT2 = typing.TypeVar("DT2", bound=DataPoint)
# def store_embeddings(
#     data_points: Union[List[DT], Dataset], storage_mode: str, dynamic_embeddings: Optional[List[str]] = None
# ):
#     if isinstance(data_points, Dataset):
#         data_points = list(_iter_dataset(data_points))

#     # if memory mode option 'none' delete everything
#     if storage_mode == "none":
#         dynamic_embeddings = None

#     # if dynamic embedding keys not passed, identify them automatically
#     elif dynamic_embeddings is None:
#         dynamic_embeddings = identify_dynamic_embeddings(data_points)

#     # always delete dynamic embeddings
#     for data_point in data_points:
#         data_point.clear_embeddings(dynamic_embeddings)

#     # if storage mode is "cpu", send everything to CPU (pin to memory if we train on GPU)
#     if storage_mode == "cpu":
#         pin_memory = str(flair.device) != "cpu"
#         for data_point in data_points:
#             data_point.to("cpu", pin_memory=pin_memory)


tag_dict = Dictionary().load_from_file("/home/sangdt/research/intent-entity-bert/entity_recognitor/tag_dictionary/tag_dictionary.pkl")
label_dictionary = Dictionary(add_unk = False)
tag_format = "BIO"
for label in tag_dict.get_items():
    if label == "<unk>":
        continue
    label_dictionary.add_item("O")
    if tag_format == "BIOES":
        label_dictionary.add_item("S-" + label)
        label_dictionary.add_item("B-" + label)
        label_dictionary.add_item("E-" + label)
        label_dictionary.add_item("I-" + label)
    if tag_format == "BIO":
        label_dictionary.add_item("B-" + label)
        label_dictionary.add_item("I-" + label)
if not label_dictionary.start_stop_tags_are_set():
    label_dictionary.set_start_stop_tags()
# print(label_dictionary.get_idx_for_item("<STOP>"))
print(label_dictionary)
viterbi_decoder = ViterbiDecoder(label_dictionary)

def _make_padded_tensor_for_batch(embeddings, sentences: List[Sentence]) -> Tuple[torch.LongTensor, torch.Tensor]:
    names = embeddings.get_names()
    lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
    longest_token_sequence_in_batch: int = max(lengths)
    pre_allocated_zero_tensor = torch.zeros(
        embeddings.embedding_length * longest_token_sequence_in_batch,
        dtype=torch.float,
        device=flair.device,
    )
    all_embs = list()
    for sentence in sentences:
        all_embs += [emb for token in sentence for emb in token.get_each_embedding(names)]
        nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

        if nb_padding_tokens > 0:
            t = pre_allocated_zero_tensor[: embeddings.embedding_length * nb_padding_tokens]
            all_embs.append(t)

    sentence_tensor = torch.cat(all_embs).view(
        [
            len(sentences),
            longest_token_sequence_in_batch,
            embeddings.embedding_length,
        ]
    )
    return torch.LongTensor(lengths), sentence_tensor




# def get_spans_from_bio(bioes_tags: List[str], bioes_scores=None) -> List[typing.Tuple[List[int], float, str]]:
#     # add a dummy "O" to close final prediction
#     bioes_tags.append("O")
#     # return complex list
#     found_spans = []
#     # internal variables
#     current_tag_weights: Dict[str, float] = defaultdict(lambda: 0.0)
#     previous_tag = "O-"
#     current_span: List[int] = []
#     current_span_scores: List[float] = []
#     for idx, bioes_tag in enumerate(bioes_tags):
#         # non-set tags are OUT tags
#         if bioes_tag == "" or bioes_tag == "O" or bioes_tag == "_":
#             bioes_tag = "O-"

#         # anything that is not OUT is IN
#         in_span = False if bioes_tag == "O-" else True

#         # does this prediction start a new span?
#         starts_new_span = False

#         # begin and single tags start new spans
#         if bioes_tag[0:2] in ["B-", "S-"]:
#             starts_new_span = True

#         # in IOB format, an I tag starts a span if it follows an O or is a different span
#         if bioes_tag[0:2] == "I-" and previous_tag[2:] != bioes_tag[2:]:
#             starts_new_span = True

#         # single tags that change prediction start new spans
#         if bioes_tag[0:2] in ["S-"] and previous_tag[2:] != bioes_tag[2:]:
#             starts_new_span = True

#         # if an existing span is ended (either by reaching O or starting a new span)
#         if (starts_new_span or not in_span) and len(current_span) > 0:
#             # determine score and value
#             span_score = sum(current_span_scores) / len(current_span_scores)
#             span_value = max(current_tag_weights.keys(), key=current_tag_weights.__getitem__)

#             # append to result list
#             found_spans.append((current_span, span_score, span_value))

#             # reset for-loop variables for new span
#             current_span = []
#             current_span_scores = []
#             current_tag_weights = defaultdict(lambda: 0.0)

#         if in_span:
#             current_span.append(idx)
#             current_span_scores.append(bioes_scores[idx] if bioes_scores else 1.0)
#             weight = 1.1 if starts_new_span else 1.0
#             current_tag_weights[bioes_tag[2:]] += weight

#         # remember previous tag
#         previous_tag = bioes_tag

#     return found_spans




onnx_model_path = "/home/sangdt/research/intent-entity-bert/entity_recognitor/flert-embeddings_2.onnx"
input_texts = ["I want to book this hotel"]


base_embeddings = TransformerWordEmbeddings(
            "bert-base-uncased",
            layers="-1,-2,-3,-4",
            layer_mean=False,
            allow_long_sentences=True,
            force_device=torch.device("cpu")
        )


# get inputs
sentences = [Sentence("to speak to a customer service advisor"), Sentence("to speak to a customer")]
tensors = base_embeddings.prepare_tensors(sentences)
# print(tensors)


# run onnx session
sess = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'] )
# input_name0 = sess.get_inputs()[0].name
# input_name1 = sess.get_inputs()[1].name
inputs = [x.name for x in sess.get_inputs()]
# ['input_ids', 'token_lengths', 'attention_mask', 'overflow_to_sample_mapping', 'word_ids']
# get output
outputs = [x.name for x in sess.get_outputs()]
# ['token_embeddings']
# print(inputs)
# print(outputs)


# start = time.time()
result1 = sess.run(outputs, {k: v.cpu().numpy() for k, v in tensors.items()})
print(len(result1))
print(result1[0].shape)
print(type(result1[0]))
print(result1[0].shape[0], type(result1[0].shape[0]))


# --------------------------------------------------------------
import os
# import test_torch
import onnx
import onnxruntime

onnx_model_path = "/home/sangdt/research/intent-entity-bert/entity_recognitor/dyn_sequencetagger2.onnx"
model = onnx.load(onnx_model_path)
sess2 = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'] )
input_name0 = sess2.get_inputs()[0].name
input_name1 = sess2.get_inputs()[1].name
inputs = [x.name for x in sess2.get_inputs()]
outputs = [x.name for x in sess2.get_outputs()]

print("--------------------------")
print(inputs)
print(sess2.get_inputs()[0].shape)
print(sess2.get_inputs()[1].shape)

print(outputs)

print(sess2.get_outputs()[0].shape)
print(sess2.get_outputs()[1].shape)
print(sess2.get_outputs()[2].shape)

# length1 = np.array([result1[0].shape[1]])
# print(length1.shape, length1)
lengths = np.array([len(sentence.tokens) for sentence in sentences])


features = sess2.run(outputs, {input_name0:result1[0],
                                input_name1: lengths})
# print(features.shape)
 
for sentence in sentences:
    sentence.remove_labels('ner')

return_probabilities_for_all_classes = False
predictions, all_tags = viterbi_decoder.decode(
                        features, return_probabilities_for_all_classes, sentences
                    )

print(predictions)

# print(all_tags)
predict_spans = True
# predict
force_token_predictions = False
label_name = 'ner'

for sentence, sentence_predictions in zip(sentences, predictions):
    # BIOES-labels need to be converted to spans
    if predict_spans and not force_token_predictions:
        sentence_tags = [label[0] for label in sentence_predictions]
        sentence_scores = [label[1] for label in sentence_predictions]
        predicted_spans = get_spans_from_bio(sentence_tags, sentence_scores)
        for predicted_span in predicted_spans:
            span: Span = sentence[predicted_span[0][0] : predicted_span[0][-1] + 1]
            span.add_label(label_name, value=predicted_span[2], score=predicted_span[1])

    # token-labels can be added directly ("O" and legacy "_" predictions are skipped)
    else:
        for token, label in zip(sentence.tokens, sentence_predictions):
            if label[0] in ["O", "_"]:
                continue
            token.add_label(typename=label_name, value=label[0], score=label[1])

# all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
for sentence, sent_all_tags in zip(sentences, all_tags):
    for token, token_all_tags in zip(sentence.tokens, sent_all_tags):
        token.add_tags_proba_dist(label_name, token_all_tags)

embedding_storage_mode = 'none'
store_embeddings(sentences, storage_mode=embedding_storage_mode)


for sentence in sentences:
    for entity in sentence.get_spans('ner'):
        print(f"Text: {sentence}")
        print(f"Entity: {entity} - {entity.tag}")