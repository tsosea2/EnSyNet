from absl import app
from absl import flags
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tqdm import tqdm

import json
import os
import pandas as pd
import numpy as np
import random
import torch

device = torch.device("cuda")

FLAGS = flags.FLAGS
flags.DEFINE_boolean("mixed_precision", True, "")

flags.DEFINE_integer("batch_size", 64, "")
flags.DEFINE_integer("max_length", 128, "")
flags.DEFINE_integer("max_epochs", 1, "")
flags.DEFINE_integer("seed", 1, "")

flags.DEFINE_string("csv_path", "data/encouragement.csv", "")
flags.DEFINE_string("results_file", "encouragement.json", "")
flags.DEFINE_string("model_type", "distilbert-base-uncased",
                    "Model Name from HuggingFace")


class SupportModel(torch.nn.Module):
    def __init__(self, ckpt_file):
        super().__init__()
        self.config = AutoConfig.from_pretrained(ckpt_file)
        self.model = AutoModel.from_pretrained(ckpt_file)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(768, 1)

    def forward(self, x):
        out = self.model(
            x['input_ids'], x['attention_mask']).last_hidden_state[:, 0, :]
        out = torch.squeeze(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = torch.squeeze(out)
        return out


def evaluate(model, test_dataloader, criterion):
    full_predictions = []
    true_labels = []

    model.eval()

    for elem in test_dataloader:
        x = {key: elem[key].to(device)
             for key in elem if key not in ['text', 'idx']}
        logits = model(x)
        results = (logits > 0.5).type(torch.LongTensor)

        full_predictions = full_predictions + \
            list(results.cpu().detach().numpy())
        true_labels = true_labels + list(elem['labels'].cpu().detach().numpy())

    model.train()

    return f1_score(true_labels, full_predictions), precision_score(true_labels, full_predictions), recall_score(true_labels, full_predictions)


def create_splits(filepath):
    df = pd.read_csv(filepath)

    texts, labels = list(df['Text']), list(df['Label'])
    X_train, X_tv, y_train, y_tv = train_test_split(
        texts, labels, test_size=0.2, random_state=42, shuffle=True)
    X_dev, X_test, y_dev, y_test = train_test_split(X_tv, y_tv, test_size=0.5, random_state=42)
    return X_train, y_train, X_dev, y_dev, X_test, y_test


class SupportDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, labels, tokenizer):
        self.text_list = text_list
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tok = self.tokenizer(
            self.text_list[idx], padding='max_length', max_length=128, truncation=True)
        item = {key: torch.tensor(tok[key]) for key in tok}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item

    def __len__(self):
        return len(self.labels)


def main(argv):

    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    if os.path.exists(FLAGS.results_file):
      with open(FLAGS.results_file) as f:
        d = json.load(f)
    else:
      d = {}

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_type, use_fast=True)
    X_train, y_train, X_dev, y_dev, X_test, y_test = create_splits(
        FLAGS.csv_path)
    ds_train, ds_dev, ds_test = SupportDataset(X_train, y_train, tokenizer), SupportDataset(
        X_dev, y_dev, tokenizer), SupportDataset(X_test, y_test, tokenizer)
    
    train_dataloader, dev_dataloader, test_dataloader = torch.utils.data.DataLoader(
        ds_train, batch_size=FLAGS.batch_size, shuffle=True), torch.utils.data.DataLoader(
        ds_dev, batch_size=FLAGS.batch_size, shuffle=True), torch.utils.data.DataLoader(
        ds_test, batch_size=FLAGS.batch_size, shuffle=True)

    model = SupportModel(FLAGS.model_type)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    if FLAGS.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    best_idx = 0
    best_f1 = 0
    for epoch in range(FLAGS.max_epochs):
        for data in tqdm(train_dataloader):
            cuda_tensors = {key: data[key].to(
                device) for key in data if key not in ['text', 'idx']}

            optimizer.zero_grad()
            if FLAGS.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = model(cuda_tensors)
                    loss = loss_fn(logits, cuda_tensors['labels'])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(cuda_tensors)
                loss = loss_fn(logits, cuda_tensors['labels'])
                loss.backward()
                optimizer.step()

        f1_validation, _, _ = evaluate(
            model, dev_dataloader, criterion)
        f1_test, precision_test, recall_test = evaluate(
            model, test_dataloader, criterion)
        print(f1_validation, f1_test)

        if f1_validation > best_f1:
            best_f1 = f1_validation
            best_precision = precision_test
            best_recall = recall_test
            best_idx = epoch
            corresponding_test = f1_test

    if str(FLAGS.seed) + '_' + FLAGS.model_type not in d:
      d[str(FLAGS.seed) + '_' + FLAGS.model_type] = {}
      
    d[str(FLAGS.seed) + '_' + FLAGS.model_type]['f1'] = best_f1
    d[str(FLAGS.seed) + '_' + FLAGS.model_type]['precision'] = best_precision
    d[str(FLAGS.seed) + '_' + FLAGS.model_type]['recall'] = best_recall
    d[str(FLAGS.seed) + '_' + FLAGS.model_type]['epoch'] = best_idx

    with open(FLAGS.results_file, 'w') as f:
      json.dump(d, f)

if __name__ == "__main__":
    app.run(main)
