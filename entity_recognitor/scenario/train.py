from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Corpus
from torch.optim.adamw import AdamW
from flair.trainers import ModelTrainer
from flair.training_utils import AnnealOnPlateau, Optimizer

from dataloader import ner_dataset

class Trainer:

    def __init__(self, args):
        self.args = args

        self.corpus, self.tag_dictionary = ner_dataset(args)

        self.embeddings = TransformerWordEmbeddings(model=args.embedding_name,
                                       layers="-1",
                                       subtoken_pooling="first",
                                       fine_tune=True,
                                       use_context=True)

        self.tagger = SequenceTagger(hidden_size=args.hidden_size,
                                embeddings=self.embeddings,
                                tag_dictionary=self.tag_dictionary,
                                tag_type=args.tag_type,
                                use_rnn=True,
                                reproject_embeddings=True,
                                tag_format=args.tag_format,
                                use_crf=True)

        self.trainer = ModelTrainer(self.tagger, self.corpus)

        self.scheduler = AnnealOnPlateau
        self.optimizer = AdamW

    def fit(self):
        self.trainer.train(self.args.checkpoints_path,
                           learning_rate=self.args.learning_rate,
                           mini_batch_size=self.args.batch_size,
                           max_epochs=self.args.num_epochs,
                           scheduler=self.scheduler,
                           optimizer=self.optimizer)

def train(args):
    trainer = Trainer(args)
    trainer.fit()