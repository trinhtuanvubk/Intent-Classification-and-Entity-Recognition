import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pickle
from sklearn.preprocessing import LabelEncoder


def infer(args):
    if isinstance(args.sentence, str):
        input_texts = [args.sentence]
    else:
        input_texts = args.sentence

    with open('./label_encoder/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    ckpt = args.checkpoints_path
    # ckpt = os.path.join(args.checkpoints_path, "model.pt")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(ckpt, local_files_only=True ).to(args.device)

    encoded_inputs = tokenizer(input_texts, padding='max_length', truncation=True, return_tensors='pt')

    input_ids = encoded_inputs['input_ids'].to(args.device)
    attention_mask = encoded_inputs['attention_mask'].to(args.device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        _, predicted_indices = torch.max(logits, dim=1)
        predicted_labels = label_encoder.inverse_transform(predicted_indices.cpu().numpy())
    # output_texts = predicted_labels.tolist()
    for text, pred in zip(input_texts, predicted_labels):
        print(f"Text: {text} - Predicted: {pred}")

    