"""
Minimal QLoRA Supervised Fine-Tuning script.
Usage:
    python src/training/train_sft.py configs/sft_llama3_colab.yml
"""
from pathlib import Path
import yaml, pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def build_prompt(ex):
    return f"<|system|>You are a helpful math tutor.<|end|>\n<|user|>{ex['question']}<|end|>\n<|assistant|>{' '.join(ex['solution_steps'])}"

def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    # === Data ===
    df = pd.read_parquet(cfg["dataset_path"])
    ds = Dataset.from_pandas(df)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tok.pad_token = tok.eos_token

    def tok_fn(batch):
        return tok([build_prompt(x) for x in batch["__index_level_0__"]],
                   truncation=True, max_length=cfg["max_seq_len"])
    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    # === Model ===
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        load_in_8bit=cfg["load_in_8bit"],
        trust_remote_code=True,
        device_map="auto"
    )
    if cfg["load_in_8bit"]:
        lora = LoraConfig(
            r=64, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
        )
        model = get_peft_model(model, lora)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    args = TrainingArguments(
        cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation"],
        learning_rate=cfg["lr"],
        warmup_steps=cfg["warmup_steps"],
        fp16=not cfg["load_in_8bit"],
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(model, args, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(cfg["output_dir"] + "/final")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
