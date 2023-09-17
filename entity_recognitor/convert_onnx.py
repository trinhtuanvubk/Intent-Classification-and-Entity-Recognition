
import torch
import transformers
from flair.data import Sentence

from flair.models import SequenceTagger


from flair.embeddings import TransformerWordEmbeddings


def convert_sequence_tagger(model_path):
    model = SequenceTagger.load(model_path)
    example_sentence = Sentence("This is a sentence.")
    longer_sentence = Sentence("This is a way longer sentence to ensure varying lengths work with LSTM.")

    reordered_sentences = sorted([example_sentence, longer_sentence], key=len, reverse=True)
    # rnn paded need order sentences
    tensors = model._prepare_tensors(reordered_sentences)
    # 
    print(tensors[0].shape)
    print(tensors[1].shape)

    print(len(tensors))
    torch.onnx.export(
        model,
        tensors,
        "dyn_sequencetagger2.onnx",
        input_names=["sentence_tensor", "lengths.1"],
        output_names=["scores", "lengths", "319"],
        dynamic_axes={"sentence_tensor" : {0: 'batch_size',
                                           1: 'max_length'},
                    "lengths.1" : {0: 'batch_size'},
                    "scores" : {0: 'batch_size',
                                1: 'max_length',
                                2: 'num_tags',
                                3: 'num_tags'},
                    "lengths" : {0: 'batch_size'},
                    "319": {0: 'num_tags',
                            1: 'num_tags'}},
        opset_version=9,
        verbose=True,
    )


def convert_sequence_tagger_2(model_path):
    model = SequenceTagger.load(model_path)
    base_embeddings = TransformerWordEmbeddings(
            "bert-base-uncased",
            layers="-1,-2,-3,-4",
            layer_mean=False,
            allow_long_sentences=True,
            force_device=torch.device("cpu")
        )


    # get inputs
    sentences = [Sentence("to speak to a customer service advisor")]
    tensors = base_embeddings.prepare_tensors(sentences)
        
    # print(tensors[0].shape)
    # print(tensors[1].shape)

    print(len(tensors))
    torch.onnx.export(
        model,
        tensors,
        "dyn_full_sequencetagger.onnx",
        input_names=['input_ids', 'token_lengths', 'attention_mask', 'overflow_to_sample_mapping', 'word_ids'],
        output_names=["scores"],
        dynamic_axes={"input_ids" : {0: 'batch_size',
                                           1: 'max_length'},
                    "attention_mask" : {0: 'batch_size',
                                           1: 'max_length'},
                    "overflow_to_sample_mapping": {0: 'max_length'},
                    "token_lengths": {0: 'max_length'},
                    "word_ids" : {0: 'batch_size',
                                    1: 'max_length'},

                    "scores" : {0: 'batch_size',
                                1: 'max_length'}},
        opset_version=9,
        verbose=True,
    )


if __name__=="__main__":
    model_path = "./checkpoints/best-model.pt"
    # model_path = "/home/sangdt/research/intent-entity-bert/entity_recognitor/checkpoints/best-model.pt"
    
    # base_embeddings = TransformerWordEmbeddings(
    #         "bert-base-uncased",
    #         layers="-1,-2,-3,-4",
    #         layer_mean=False,
    #         allow_long_sentences=True,
    #         force_device=torch.device("cpu")
    #     )

    # convert_sequence_tagger(model_path)

    convert_sequence_tagger_2(model_path)
