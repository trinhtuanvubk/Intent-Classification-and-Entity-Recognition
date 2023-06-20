from torch.utils.data import DataLoader
import json
from sklearn.preprocessing import LabelEncoder
import pickle
from .dataset import CustomDataset


def intent_loader(args, tokenizer):
    label_encoder = LabelEncoder()

    with open(args.train_path) as f:
        train_data = [json.loads(line) for line in f]
    with open(args.valid_path) as f:
        valid_data = [json.loads(line) for line in f]

    total_data = train_data + valid_data
    unique_intentions = set(item['intention'] for item in total_data)
    num_labels = len(unique_intentions)

    # get label
    train_intentions = [item['intention'] for item in train_data]
    valid_intentions = [item['intention'] for item in valid_data]
    all_intentions = train_intentions + valid_intentions

    label_encoder.fit(all_intentions)
    train_labels = label_encoder.transform(train_intentions)
    valid_labels = label_encoder.transform(valid_intentions)

    train_dataset = train_dataset = CustomDataset(train_data, train_labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    valid_dataset = CustomDataset(valid_data, valid_labels, tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # Save the label encoder
    os.makedirs("./label_encoder/", exist_ok=True)
    with open('label_encoder/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    return train_dataloader, valid_dataloader, num_labels


