# Team CodeCraft Members :
# 1. Sumedh Ghavat (115854819)
# 2. Tanmay Armal (115970117)
# 3. Shreyas Habade (115911132)


# The code uses Code Llama to generate the python code as responses as potential solutions to the leetcode problems.
# The model is trained in autoregressive nature by giving problem statement/content and python gold solution for training
# The model is then evaluated using BLEU score, which measures the similarity between a generated sentence and a reference sentence.

# 4 concepts used:
#   1. Syntax - Tokenization
#   2. Semantics - Dependency Parsing in the content and the code
#   3. Language Modeling - Using Code-Llama as the model and fine tuning, doing language modeling
#   4. Applications - Question Answering as context is given as promt and code is given out as answer.

# !pip install transformers datasets torch sacrebleu
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, LlamaForCausalLM, CodeLlamaTokenizer

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sacrebleu.metrics import BLEU

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Datasets

# Loading the dataset
dataset = load_dataset("greengerong/leetcode")

# Creating train and test split
dataset = dataset["train"].train_test_split(test_size=0.2)
train_dataset = dataset['train']
test_dataset = dataset['test']

# Loading tokenizer and model
# model_name = "allenai/llama-13b-code"  # Replace this with the actual model suitable for the task
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# preprocessing function for concatenating the content and the python code
def preprocess_function(examples):
    inputs = ["solve: " + problem + " [SEP] " for problem in examples["content"]]  # Separator token [SEP]
    targets = [code for code in examples["python"]]
    concatenated_examples = [inp + target for inp, target in zip(inputs, targets)]

    model_inputs = tokenizer(concatenated_examples, max_length=640, truncation=True, padding="max_length")

    # Ensure that labels are properly aligned with the inputs for the autoregressive task
    labels = model_inputs.input_ids.copy()
    for i, (input_ids, label) in enumerate(zip(model_inputs.input_ids, labels)):
        labels[i] = [-100 if token == tokenizer.pad_token_id else token for token in label]

    model_inputs["labels"] = labels
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Training

# model_name = "allenai/llama-13b-code"  # Example model name for coding tasks
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)
trainer.train()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# BLEU Score Evaluation

bleu = BLEU()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate BLEU score
    bleu_score = bleu.corpus_score(decoded_preds, [decoded_labels])
    return {"bleu": bleu_score.score}

# Add evaluation function to trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluate the model
results = trainer.evaluate()
print(results)