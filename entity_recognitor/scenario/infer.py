import os
from flair.data import Sentence
from flair.models import SequenceTagger


def infer(args):
    # ckpt = os.path.join(args.checkpoints_path, "best-model.pt")
    # print(ckpt)
    ckpt = f"{args.checkpoints_path}/best-model.pt"
    model = SequenceTagger.load(ckpt)
    # create example sentence
    sentence = Sentence(args.sentence)
    # predict the tags
    model.predict(sentence)
    for entity in sentence.get_spans('ner'):
        print(f"Text: {sentence}")
        print(f"Entity: {entity} - {entity.tag}")


