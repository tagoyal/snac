import argparse
import logging

import numpy as np
import torch
from torch import nn
import os
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm.auto import tqdm
import csv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    SchedulerType,
    get_scheduler,
    set_seed,
)

logger = logging.getLogger(__name__)
softmax_func = nn.Softmax(dim=1)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    args = parser.parse_args()
    return args


def load_dataset(data_file):
    reader = csv.DictReader(open(data_file), delimiter='\t')
    examples = [row for row in reader]
    return examples


def evaluate(model, eval_dataloader, tokenizer, args):
    model.eval()

    f_out = open(os.path.join(args.output_dir, 'dev_out.txt'), 'w')

    for step, batch in tqdm(enumerate(eval_dataloader), desc='Evaluation'):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, attention_mask = batch[0], batch[1]
        inp = {'input_ids': input_ids, 'attention_mask': attention_mask}
        with torch.no_grad():
            outputs = model.generate(**inp, max_length=80)
            predictions_gen = [tokenizer.decode(outputs[i], skip_special_tokens=False) for i in range(outputs.shape[0])]

            for inp, p in zip(input_ids, predictions_gen):
                inp = tokenizer.decode(inp, skip_special_tokens=False).replace('<pad>', '').strip()
                p = p.replace('<pad>', '').strip()
                f_out.write(inp + '\n')
                f_out.write(f'Predicted: {str(p)}\n')


def save_model(model, tokenizer, args, output_dir):
    model_to_save = (model.module if hasattr(model, "module") else model)
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


def main():
    args = parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    val_data_raw = load_dataset(args.validation_file)
    print(len(val_data_raw))

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    args.device = 'cuda'
    model.to(args.device)

    def process_data(data_raw):
        texts = []
        for row_idx, row in enumerate(data_raw):
            print(row)
            text_temp = ' '.join(row['context'].split()[-int(0.9 * args.max_length):]) + \
                        ' <extra_id_0> ' + row['sentence']
            texts.append(text_temp)

        texts = tokenizer(texts, max_length=args.max_length, truncation=True,
                          padding='max_length', return_tensors='pt')

        input_ids, attention_mask = texts.input_ids, texts.attention_mask
        dataset = TensorDataset(input_ids, attention_mask)
        return dataset

    eval_dataset = process_data(val_data_raw)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size)

    evaluate(model, eval_dataloader, tokenizer, args)

if __name__ == "__main__":
    main()
