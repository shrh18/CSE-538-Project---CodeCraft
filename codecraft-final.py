from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from codebleu import calc_codebleu
from nltk.translate.bleu_score import corpus_bleu
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os
import matplotlib.pyplot as plt

# Unique identifiers
HF_TOKEN = "your_own_hf_token_here"
MODEL_ID = "codellama/CodeLlama-7b-hf"
FINETUNED_MODEL_ID = "sumedhghavat/codellama2-finetuned-codex"
FINETUNED_MODEL_NAME = "codellama2-finetuned-codex-fin"

def load_and_customize_dataset():
    # Load dataset
    data = load_dataset("greengerong/leetcode", split="train")
    data_df = data.to_pandas()

    # Customize data for training
    data_df["text"] = data_df[["content", "python"]].apply(lambda x: "[INFO] Problem: " + x["content"] + " [/INFO] Code: " + x["python"] + "", axis=1)
    return Dataset.from_pandas(data_df)

def tokenize_data():
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def initialize_model():
    # Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )

    # Model initialization
    model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-hf", quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model

def configure_training():
    # Training configuration
    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    training_arguments = TrainingArguments(
        output_dir="custom-finetuned-model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=1,
        max_steps=100,
        fp16=True,
        push_to_hub=True
    )
    return peft_config, training_arguments

def initialize_trainer(model, data, peft_config, tokenizer):
    # Trainer initialization
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=512
    )
    return trainer

def train_and_push_to_hub(trainer):
    # Start training
    trainer.train()

    # Push the finetuned model to the hub
    trainer.push_to_hub()

def custom_finetune():
    data = load_and_customize_dataset()
    tokenizer = tokenize_data()
    model = initialize_model()
    peft_config, training_arguments = configure_training()
    trainer = initialize_trainer(model, data, peft_config, tokenizer)
    train_and_push_to_hub(trainer)

def initialize_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, load_in_8bit=False, device_map="auto", trust_remote_code=True
    )
    return model

def load_and_merge_finetuned_model(model):
    peft_model = PeftModel.from_pretrained(model, FINETUNED_MODEL_ID, from_transformers=True, device_map={"":0})
    merged_model = peft_model.merge_and_unload()
    return merged_model

def push_model_to_hub(model, model_name):
    model.push_to_hub(model_name)

def tokenize_and_push_to_hub():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.push_to_hub(FINETUNED_MODEL_NAME)
    return tokenizer

def initialize_text_generation_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_NAME)
    pipe = pipeline(
        "text-generation",
        model=FINETUNED_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, pipe

def generate_text_sequences(tokenizer, pipe):
    sequences = pipe(
        'def fibonacci(',
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=100,
    )
    return sequences

def print_generated_sequences(sequences):
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

def load_and_preprocess_dataset(split="train"):
    # Load dataset
    dataset = load_dataset("greengerong/leetcode", split=split)
    data_df = dataset.to_pandas()

    # Customize data for training/validation
    data_df["text"] = data_df[["content", "python"]].apply(lambda x: "[INFO] Problem: " + x["content"] + " [/INFO] Code: " + x["python"] + "", axis=1)

    return Dataset.from_pandas(data_df)

def calculate_metrics(validation_sequences, tokenizer):
    bleu_scores = []
    codebleu_scores = []

    # Reference for BLEU score
    references = validation_sequences

    for sequence in validation_sequences:
        # Generate text using the pipeline
        generated_text = text_generation_pipe(sequence)

        # Tokenize the generated text
        generated_tokens = tokenizer.tokenize(generated_text)

        # Tokenize the reference text
        reference_tokens = tokenizer.tokenize(sequence)

        # Calculate BLEU score
        bleu_score = corpus_bleu([[reference_tokens]], [generated_tokens])
        bleu_scores.append(bleu_score)

        # Calculate CodeBLEU score
        codebleu_score = calc_codebleu([sequence], [generated_text], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=tokenizer)
        codebleu_scores.append(codebleu_score['codebleu'])

    return bleu_scores, codebleu_scores


def plot_bleu_scores_difficulty_updated(bleu_scores=list(), difficulty_levels=["Easy", "Medium", "Hard"]):
    # Plot BLEU Score Graph (Updated)
    plt.figure(figsize=(10, 5))
    plt.plot(difficulty_levels, bleu_scores, marker='o', linestyle='-', color='blue', label='BLEU Score')
    plt.title('BLEU Scores by Difficulty (Updated)')
    plt.xlabel('Problem Difficulty')
    plt.ylabel('BLEU Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('/mnt/data/bleu_scores_difficulty_updated_matplotlib.png')
    plt.show()

def plot_codebleu_scores_difficulty_updated(codebleu_scores=list(), difficulty_levels=["Easy", "Medium", "Hard"]):
    # Plot CodeBLEU Score Graph (Updated)
    plt.figure(figsize=(10, 5))
    plt.bar(difficulty_levels, codebleu_scores, color=['green', 'orange', 'red'], alpha=0.7)
    plt.title('CodeBLEU Scores by Difficulty (Updated)')
    plt.xlabel('Problem Difficulty')
    plt.ylabel('CodeBLEU Score')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('/mnt/data/codebleu_scores_difficulty_updated_matplotlib.png')
    plt.show()

def plot_training_loss_curve(training_data=list()):
    # Extract loss and epoch data
    epochs = [entry['epoch'] for entry in training_data]
    losses = [entry['loss'] for entry in training_data]

    # Plot training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('/mnt/data/training_loss_curve_matplotlib.png')
    plt.show()

def plot_token_distribution_difficulty(difficulty_levels, average_token_counts):
    # Plot token distribution graph
    plt.figure(figsize=(10, 5))
    plt.bar(difficulty_levels, average_token_counts, color=['green', 'orange', 'red'], alpha=0.7)
    plt.title('Average Token Distribution by Problem Difficulty')
    plt.xlabel('Problem Difficulty')
    plt.ylabel('Average Token Count')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('/mnt/data/token_distribution_difficulty_matplotlib.png')
    plt.show()


def main():
    # Set environment variable
    os.environ["HF_TOKEN"] = HF_TOKEN

    # Initialize base model
    base_model = initialize_base_model()

    # Load and merge finetuned model
    merged_model = load_and_merge_finetuned_model(base_model)

    # Push finetuned model to hub
    push_model_to_hub(merged_model, FINETUNED_MODEL_NAME)

    # Tokenize and push tokenizer to hub
    tokenizer = tokenize_and_push_to_hub()

    # perform finetuning
    custom_finetune()

    # Initialize text generation pipeline
    tokenizer, text_generation_pipe = initialize_text_generation_pipeline()

    # Generate text sequences
    sequences = generate_text_sequences(tokenizer, text_generation_pipe)

    # Generate text sequences for validation
    validation_sequences = generate_text_sequences(tokenizer, text_generation_pipe)

    # Print generated sequences for validation
    print("\nValidation Sequences:")
    print_generated_sequences(validation_sequences)


    # Calculate BLEU and CodeBLEU metrics
    bleu_scores, codebleu_scores = calculate_metrics(validation_sequences, tokenizer)

    # Print BLEU and CodeBLEU scores for each sequence
    print("Validation BLEU Scores:")
    for i, score in enumerate(bleu_scores):
        print(f"Sequence {i+1}: BLEU = {score}")

    print("\nValidation CodeBLEU Scores:")
    for i, score in enumerate(codebleu_scores):
        print(f"Sequence {i+1}: CodeBLEU = {score}")

    # Function 1: Plot BLEU Scores by Difficulty (Updated)
    plot_bleu_scores_difficulty_updated(bleu_scores, difficulty_levels)

    # Function 2: Plot CodeBLEU Scores by Difficulty (Updated)
    plot_codebleu_scores_difficulty_updated(codebleu_scores, difficulty_levels)

    plot_token_distribution_difficulty(difficulty_levels, average_token_counts)

# Execute main function
if __name__ == "__main__":
    main()