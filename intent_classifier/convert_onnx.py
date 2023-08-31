import torch
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification

device = torch.device("cuda")
ckpt = "./checkpoints/"
onnx_path = "./checkpoints/model.onnx"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained(ckpt, local_files_only=True ).to(device)


dummy_model_input = tokenizer("I want to book this hotel", padding='max_length', truncation=True, return_tensors="pt")

input_ids = dummy_model_input['input_ids'].to(device)
attention_mask = dummy_model_input['attention_mask'].to(device)

# export
torch.onnx.export(
    model, 
    tuple([input_ids, attention_mask]),
    f=onnx_path,  
    verbose=True,
    input_names=['input_ids', 'attention_mask'], 
    output_names=['logits'], 
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                  'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                  'logits': {0: 'batch_size', 1: 'sequence'}}, 
    do_constant_folding=True, 
)