
import argparse
import torch

def get_args():

    parser = argparse.ArgumentParser(description='Train Intent Classification Model')
    parser.add_argument('--scenario', type=str, default='train')
    parser.add_argument('--train_path', type=str, default="./data/train_intents.json", help='Path to the training data')
    parser.add_argument('--valid_path', type=str, default="./data/valid_intents.json", help='Path to the validation data')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--model_name', type=str, default="roberta-base", help='Model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--shuffle', action="store_true", help='Shuffle data')
    parser.add_argument('--checkpoints_path', type=str, default="./checkpoints/", help='Shuffle data')


    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args