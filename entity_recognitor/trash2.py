from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence, Dictionary
import flair
import os
import torch
from typing import cast
# import test_torch
from typing import List, Tuple
import onnx
import onnxruntime
import numpy as np
from viterbi import ViterbiDecoder
# init embedding
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

embedding = TransformerWordEmbeddings('bert-base-uncased', allow_long_sentences=True)

# create a sentence
# sentence = Sentence('to speak to a customer service advisor')
sentences = [Sentence('to speak to a customer service advisor'), Sentence('to speak to a customer service')]
# embed words in sentence
Sentence.set_context_for_sentences(cast(List[Sentence], sentences))
embedding.embed(sentences)

# embed = sentence[1].embedding
lengths, sentence_tensor = _make_padded_tensor_for_batch(embedding, sentences)
print(lengths, sentence_tensor)
print(lengths.shape, sentence_tensor.shape) 
print(embedding.get_names())
print(embedding.embedding_length)

# full_embed = np.array([i.embedding.cpu().numpy() for i in sentence])



onnx_model_path = "/home/sangdt/research/intent-entity-bert/entity_recognitor/dyn_sequencetagger2.onnx"
# model = onnx.load(onnx_model_path)
sess = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'] )

input_name0 = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
inputs = [x.name for x in sess.get_inputs()]
outputs = [x.name for x in sess.get_outputs()]
print(inputs)
print(outputs)
print([x.shape for x in sess.get_inputs()])
print([x.shape for x in sess.get_outputs()])

print(np.array(lengths.cpu().numpy()))
# logits = sess.run(outputs, {input_name0: np.expand_dims(full_embed, 0)},
                            # input_name1: )

features = sess.run(outputs, {input_name0: np.array([t.cpu().numpy() for t in sentence_tensor]),
                            input_name1: np.array(lengths.cpu().numpy())})

# print(features.shape)
# print(features[0])
# print(features[1])
# print(features[2])

for sentence in sentences:
    sentence.remove_labels('ner')


return_probabilities_for_all_classes = False
predictions, all_tags = viterbi_decoder.decode(
                        features, return_probabilities_for_all_classes, sentences
                    )

print(predictions)

# print(all_tags)

for sentence, sentence_predictions in zip(batch, predictions):
    # BIOES-labels need to be converted to spans
    if self.predict_spans and not force_token_predictions:
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

store_embeddings(sentences, storage_mode=embedding_storage_mode)
