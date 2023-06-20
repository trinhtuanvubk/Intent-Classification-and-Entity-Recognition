
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Train Ner Model')
    parser.add_argument('--scenario', type=str, default='train')
    parser.add_argument('--tag_type', type=str, default='ner')
    parser.add_argument('--tag_format', type=str, default='BIO')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--train_path', type=str, default="./data/train_slots.json", help='Path to the training data')
    parser.add_argument('--valid_path', type=str, default="./data/valid_slots.json", help='Path to the validation data')
    parser.add_argument('--test_path', type=str, default="./data/valid_slots.json", help='Path to the test data')
    parser.add_argument('--converted_train_path', type=str, default="./data/train_slots.txt", help='Path to the training data')
    parser.add_argument('--converted_valid_path', type=str, default="./data/valid_slots.txt", help='Path to the validation data')
    parser.add_argument('--converted_test_path', type=str, default="./data/test_slots.txt", help='Path to the test data')
    parser.add_argument('--slots_path', type=str, default="./data/slots.txt", help='Path to the slots list')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--model_name', type=str, default="bert-base-uncased", help='Model name')
    parser.add_argument('--embedding_name', type=str, default="bert-base-uncased", help='embedding name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoints_path', type=str, default="./checkpoints/", help='checkpoints path')
    parser.add_argument('--sentence', type=str, default="to check my bill", help='sentence test for inference')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args