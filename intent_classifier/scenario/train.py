import argparse
import json

import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from dataloader import intent_loader

class Trainer:

    def __init__(self, args):
        self.args = args
        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
        self.train_loader, self.val_loader, self.num_labels = intent_loader(args, self.tokenizer)
        self.model = RobertaForSequenceClassification.from_pretrained(
                                                      args.model_name, 
                                                      num_labels=self.num_labels).to(args.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def fit(self):
        for epoch in range(self.args.num_epochs):
            pbar = tqdm(self.train_loader)
            self.model.train()
            total_loss = 0
            for batch in pbar:
                input_ids, attention_mask, label = batch

                input_ids = input_ids.to(self.args.device)
                attention_mask = attention_mask.to(self.args.device)
                label = label.to(self.args.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=label)
                logits = outputs.logits
                loss = self.loss_fn(logits.view(-1, self.num_labels), label.view(-1))
                pbar.set_description(f"Updating losss {loss} at epoch {epoch}")
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{epoch} - Train Loss: {average_loss:.4f}")

            # Evaluation on validation data
            self.model.eval()
            with torch.no_grad():
                valid_labels = []
                predicted_labels = []

                for batch in self.val_loader:
                    input_ids, attention_mask, label = batch
                    input_ids = input_ids.to(self.args.device)
                    attention_mask = attention_mask.to(self.args.device)

                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    _, predicted = torch.max(logits, dim=1)

                    valid_labels.extend(label.cpu().numpy())
                    predicted_labels.extend(predicted.cpu().numpy())

                valid_accuracy = accuracy_score(valid_labels, predicted_labels)
                print(f"Epoch {epoch + 1}/{epoch} - Valid Accuracy: {valid_accuracy:.4f}")

        torch.save(self.model.state_dict(), './checkpoints/model.pt')
        print("Checkpoint saved to './checkpoints/model.pt'")


def train(args):
    trainer = Trainer(args)
    trainer.fit()

