import os
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings

# from flair.datasets import CONLL_03
# sentences = list(CONLL_03().test)[:5]

# def infer(args):
    # ckpt = os.path.join(args.checkpoints_path, "best-model.pt")
    # print(ckpt)
ckpt = "./checkpoints/best-model.pt"
model = SequenceTagger.load(ckpt)
# print(dir(model.embeddings))

assert isinstance(model.embeddings, TransformerWordEmbeddings)
# create example sentence
sentences = [Sentence("to speak to a customer service advisor")]
# sentences = list(["I want to go to home", "I want to book this hotel"])

# predict the tags
# model.predict(sentence)
# for entity in sentence.get_spans('ner'):
#     print(f"Text: {sentence}")
#     print(f"Entity: {entity} - {entity.tag}")



# model.embeddings = model.embeddings.export_onnx("flert-embeddings.onnx", sentences, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
base_embeddings = TransformerWordEmbeddings(
            "bert-base-uncased", layers="-1,-2,-3,-4", layer_mean=False, allow_long_sentences=True
        )
# sentence: Sentence = Sentence("I love Berlin, but Vienna is where my hearth is.")
tensors = base_embeddings.prepare_tensors(sentences)
# tensors = tokenizer(sentences)
print(tensors)

model.predict(sentences)

for sentence in sentences:
    for entity in sentence.get_spans('ner'):
        print(f"Text: {sentence}")
        print(f"Entity: {entity} - {entity.tag}")