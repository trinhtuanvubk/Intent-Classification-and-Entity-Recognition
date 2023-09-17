from flair.datasets import ColumnCorpus
from flair.data import Corpus


def ner_dataset(args):
    # define columns
    columns = {0 : 'text', 1 : 'ner'}
    # initializing the corpus
    train_slots_filename = args.converted_train_path.split("/")[-1]
    valid_slots_filename = args.converted_valid_path.split("/")[-1]
    test_slots_filename = args.converted_test_path.split("/")[-1]
    corpus = ColumnCorpus(args.data_folder, 
                        columns,
                        train_file = train_slots_filename,
                        test_file = test_slots_filename,
                        dev_file = valid_slots_filename)

    # tag to predict
    tag_type = args.tag_type
    # make tag dictionary from the corpus
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
    alist = [line.rstrip() for line in open(args.slots_path)]
    for i in alist:
        tag_dictionary.add_item(i)
    
    tag_dictionary.save("./tag_dictionary/tag_dictionary.pkl")

    return corpus, tag_dictionary



