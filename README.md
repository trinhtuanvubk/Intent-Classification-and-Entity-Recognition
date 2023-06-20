# Intent Classification and Entity Recognition

### Setup 

- To create Docker environment:
```
docker build -t text-env .
docker run -itd --gpus all --restart always -v $(pwd)/:/workspace --name text-env text-env:latest
docker exec -it text-env bash
```

### Intent Clasification
- Using transformers framework (Huggingface) to finetune Intent Classification task that uses roberta-base as pretrained model

```bash
cd intent_classifier/
```

- For training, bring dataset to the format:
```
|data/
|   |- train_intents.json
|   |- valid_intents.json
|   |- test_intents.json
```


- To train, run command:
```
python3 main.py --scenario train \
--num_epochs 2 \
--batch_size 8 \
--learning_rate 1e-5 \
--shuffle
```

or 
```bash
bash run.sh
```

- To infer, run command:
```
python3 main.py --scenario infer \
--sentence "type your text here"
```

### Entity Recoginition
- Using Flair framework to finetune NER task that uses bert-base-uncased as pretrained model

```bash
cd entity_recognitor
```

- For training, bring dataset to the format:
```
|data/
|   |- slots.txt
|   |- train_slots.json
|   |- valid_slots.json
|   |- test_slots.json
```

- To convert data to flair format: 
```
python3 main.py --scenario convert_data
```

- To train, run command:
```
python3 main.py --scenario train \
--num_epochs 2 \
--batch_size 8 \
--learning_rate 1e-4 \
--hidden_size 256
```

or
```bash
bash run.sh
```

- To infer, run command:
```
python3 main.py --scenario infer \
--sentence "type your text here"
```