python
import torch
from datasets import DatasetDict
from transformers import TrainingArguments, Trainer, AutoModel, AutoProcessor

import os
import argparse
from datasets import load_from_disk
import random
import wandb


# Collate function for CLIP models
def collate_fn_standard_clip(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "return_loss": True
    }


# Collate function for SigLIP models
def collate_fn_standard_siglip(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "return_loss": True,
        "attention_mask": None
    }


# Set which layers to train based on arguments
def layers_to_train(model, frozen_layers=False, only_vision=False,
                    projections=False, projection_vision=False, projection_text=False):
    if frozen_layers:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "text_projection"):
            for param in model.text_projection.parameters():
                param.requires_grad = True
        if hasattr(model, "visual_projection"):
            for param in model.visual_projection.parameters():
                param.requires_grad = True
        if hasattr(model, "logit_scale"):
            model.logit_scale.requires_grad = False
    elif only_vision:
        for name, param in model.named_parameters():
            param.requires_grad = not ("text" in name)
        if hasattr(model, "logit_scale"):
            model.logit_scale.requires_grad = False
    if projections:
        if hasattr(model, "text_projection"):
            for param in model.text_projection.parameters():
                param.requires_grad = True
        if hasattr(model, "visual_projection"):
            for param in model.visual_projection.parameters():
                param.requires_grad = True
    elif projection_vision:
        if hasattr(model, "visual_projection"):
            for param in model.visual_projection.parameters():
                param.requires_grad = True
    elif projection_text:
        if hasattr(model, "text_projection"):
            for param in model.text_projection.parameters():
                param.requires_grad = True
    print("\n--- Trainable Parameters ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    model.logit_scale.requires_grad = False


# Set device and precision for training
def training_specifications(seed=42, fp16=False, bf16=False):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_fp16 = fp16 and not bf16 and torch.cuda.is_available()
    use_bf16 = bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_bf16:
        model_dtype = torch.bfloat16
        print("Using BF16 for mixed precision training.")
    elif use_fp16:
        model_dtype = torch.float16
        print("Using FP16 for mixed precision training.")
    else:
        model_dtype = torch.float32
        print("Using FP32 for training.")
    return {
        "device": device,
        "use_fp16": use_fp16,
        "use_bf16": use_bf16,
        "model_dtype": model_dtype
    }


# Main training function for a single run
def train_single_run(config=None, fixed_args=None):
    wandb.init(project=fixed_args.project_name,
               config=config,
               name=fixed_args.RUN_NAME if fixed_args else None,
               mode=fixed_args.wandb_mode if fixed_args else "online"
               )
    config = wandb.config
    try:
        num_train_epochs = config.num_train_epochs if 'num_train_epochs' in config else fixed_args.num_epochs
        per_device_train_batch_size = config.per_device_train_batch_size if 'per_device_train_batch_size' in config else fixed_args.batch_size
        per_device_eval_batch_size = fixed_args.eval_batch_size if fixed_args.eval_batch_size != -1 else per_device_train_batch_size
        learning_rate = config.learning_rate if 'learning_rate' in config else fixed_args.lr
        warmup_ratio = config.warmup_ratio if 'warmup_ratio' in config else fixed_args.warmup_ratio
        weight_decay = config.weight_decay if 'weight_decay' in config else fixed_args.weight_decay
        gradient_accumulation_steps = config.gradient_accumulation_steps if 'gradient_accumulation_steps' in config else fixed_args.gradient_accumulation
        lr_scheduler_type = config.lr_scheduler_type if 'lr_scheduler_type' in config else fixed_args.lr_scheduler
        max_grad_norm = config.max_grad_norm if 'max_grad_norm' in config else fixed_args.max_grad_norm
        trainer_type = config.trainer_type if 'trainer_type' in config else fixed_args.trainer_type
        frozen_layers = config.frozen_layers if 'frozen_layers' in config else fixed_args.frozen_layers
        only_vision = config.only_vision if 'only_vision' in config else fixed_args.only_vision
        base_model = config.base_model if 'base_model' in config else fixed_args.base_model
    except AttributeError as e:
        print(f"Error accessing config parameters: {e}")
        print("Available config keys:", list(config.keys()))
        raise

    seed = fixed_args.seed
    logging_steps = fixed_args.logging_steps
    preprocessed_dataset_dir = fixed_args.preprocessed_dataset_dir
    eval_dataset_dir = fixed_args.eval_dataset_dir
    ds_size = fixed_args.ds_size
    ds_eval_size = fixed_args.ds_eval_size
    metric_to_look_for = fixed_args.metric
    projection_layers = fixed_args.projection_layers
    projection_vision = fixed_args.projection_vision
    projection_text = fixed_args.projection_text
    RUN_BASE_PATH = ""  # Set your base path here

    if wandb.run.id:
        RUN_NAME_FOR_OUTPUT = os.path.join(RUN_BASE_PATH, wandb.run.name)
    if not os.path.exists(RUN_NAME_FOR_OUTPUT):
        os.makedirs(RUN_NAME_FOR_OUTPUT)

    print(f"Model output directory: {RUN_NAME_FOR_OUTPUT}")
    print("\n--- Training Configuration ---")
    print("WANDB Mode :", fixed_args.wandb_mode if fixed_args else "online")
    print(f"Using trainer type: {trainer_type}")
    print(f"Number of epochs: {num_train_epochs}")
    print(
        f"Batch size: {per_device_train_batch_size * gradient_accumulation_steps} (per device: {per_device_train_batch_size}, gradient accumulation: {gradient_accumulation_steps})")
    print(f"Logging steps: {logging_steps}")
    print(f"Preprocessed dataset directory: {preprocessed_dataset_dir}")
    print(f"Frozen layers: {frozen_layers}")
    print(f"Only vision parameters: {only_vision}")
    print(f"Random seed: {seed}")
    print(f"Learning rate: {learning_rate}")
    print(f"Warmup ratio: {warmup_ratio}")
    print(f"Weight decay: {weight_decay}")
    print(f"Learning rate scheduler: {lr_scheduler_type}")
    print(f"Base model: {base_model}")
    print("-" * 50)

    args_for_training = training_specifications(seed=seed, fp16=True, bf16=True)
    device = args_for_training["device"]

    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for training.")

    use_bf16 = args_for_training["use_bf16"]
    use_fp16 = args_for_training["use_fp16"]
    dtype = args_for_training["model_dtype"]

    print("Loading model...")
    model = AutoModel.from_pretrained(base_model, torch_dtype=dtype).to(device)
    print(f"Loading preprocessed dataset from {preprocessed_dataset_dir}...")

    ds_preprocessed = load_from_disk(preprocessed_dataset_dir)
    dataset_train = ds_preprocessed["train"].with_format("torch")

    # Load evaluation dataset
    if eval_dataset_dir != preprocessed_dataset_dir:
        print(f"Loading evaluation dataset from {eval_dataset_dir}...")
        ds_eval = load_from_disk(eval_dataset_dir)
        dataset_test = ds_eval.with_format("torch")
    else:
        print("Using validation split of the preprocessed dataset for evaluation.")
        dataset_test = ds_preprocessed["validation"].with_format("torch")

    # Limit dataset sizes if specified
    if ds_size > 0:
        size_to_use = min(ds_size, len(dataset_train))
        dataset_train = dataset_train.shuffle(seed=seed).select(range(size_to_use))
        print(f"Using {len(dataset_train)} samples for training (limited by --ds_size).")
    else:
        print("Using full training dataset size:", len(dataset_train))

    # Limit evaluation dataset size if specified
    if ds_eval_size > 0:
        if type(dataset_test) == DatasetDict:
            for split in dataset_test.keys():
                size_to_use = min(ds_eval_size, len(dataset_test[split]))
                dataset_test[split] = dataset_test[split].shuffle(seed=seed).select(range(size_to_use))
        else:
            size_to_use = min(ds_eval_size, len(dataset_test))
            dataset_test = dataset_test.shuffle(seed=seed).select(range(size_to_use))
        print(f"Using {len(dataset_test)} samples for evaluation (limited by --ds_eval_size).")
    else:
        print("Using full evaluation dataset size:", len(dataset_test))

    # Determine collate function based on base model
    if "openai/clip" in base_model:
        collate_fn_to_use = collate_fn_standard_clip
    elif "google/siglip" in base_model:
        collate_fn_to_use = collate_fn_standard_siglip
    else:
        raise ValueError(f"Unknown base model: {base_model}. Cannot determine collate_fn.")

    if trainer_type == "classic":
        trainer_class = Trainer
        collate_fn = collate_fn_to_use
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}.")

    print("Shuffling datasets with seed:", seed)
    dataset_train = dataset_train.shuffle(seed=seed)

    layers_to_train(model, frozen_layers=frozen_layers, only_vision=only_vision, projections=projection_layers,
                    projection_vision=projection_vision, projection_text=projection_text)

    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f"Using {num_cpus} CPU cores for multiprocessing in clean_data_for_training.")


    training_args = TrainingArguments(
        output_dir=RUN_NAME_FOR_OUTPUT,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model=metric_to_look_for,
        greater_is_better=False if "loss" in metric_to_look_for else True,
        torch_compile=True,
        remove_unused_columns=False,
        report_to="wandb",
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch_fused" if device.type == "cuda" else "adamw_torch",
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=max_grad_norm,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        dataloader_num_workers=num_cpus,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_on_start=True
    )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=collate_fn,
    )
    print("Starting training...")
    trainer.train()
    print("Training completed. Saving model...")
    trainer.save_model(RUN_NAME_FOR_OUTPUT)
    print(f"Model saved to {RUN_NAME_FOR_OUTPUT}")
    wandb.finish()


def main():
    argparser = argparse.ArgumentParser(description="Train CLIP with preprocessed data")
    argparser.add_argument("--preprocessed_dataset_dir", type=str, required=True,
                           help="Directory containing the preprocessed dataset")
    argparser.add_argument("--eval_dataset_dir", type=str, default=None,
                           help="Directory containing the evaluation dataset. If not provided, uses the validation split of the preprocessed dataset.")
    argparser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    argparser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    argparser.add_argument("--eval_batch_size", type=int, default=-1, help="Batch size for training")
    argparser.add_argument("--logging_steps", type=int, default=1, help="Logging steps for training")
    argparser.add_argument("--trainer_type", type=str, default="classic",
                           choices=["classic"],
                           help="Type of trainer to use")
    argparser.add_argument("--frozen_layers", action="store_true",
                           help="If set, freeze all layers except the logit scale and projections")
    argparser.add_argument("--only_vision", action="store_true",
                           help="If set, only train the vision parameters and freeze text parameters")
    argparser.add_argument("--projection_layers", action="store_true",
                           help="If set, train the projection layers")
    argparser.add_argument("--projection_vision", action="store_true",
                           help="If set, train the visual projection layer")
    argparser.add_argument("--projection_text", action="store_true",
                           help="If set, train the text projection layer")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    argparser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    argparser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    argparser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    argparser.add_argument("--max_grad_norm", type=float, default=1.0,
                           help="Maximum gradient norm for gradient clipping")
    argparser.add_argument("--lr_scheduler", type=str, default="cosine",
                           choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
                           help="Learning rate scheduler type")
    argparser.add_argument("--gradient_accumulation", type=int, default=1,
                           help="Number of gradient accumulation steps")
    argparser.add_argument("--base_model", type=str, default="openai/clip-vit-base-patch32",
                           help="Base CLIP model to use for training")
    argparser.add_argument("--ds_size", type=int, default=0,
                           help="Size of the training dataset to use. If 0, use full preprocessed dataset. Only applies to training data, not eval.")
    argparser.add_argument("--ds_eval_size", type=int, default=0,
                           help="Size of the evaluation dataset to use. If 0, use full eval dataset. Only applies to eval data, not training.")
    argparser.add_argument("--extra_name", type=str,
                           help="Extra name to append to the run name for clarity, e.g., 'preprocessed'")
    argparser.add_argument("--resume_training", type=str, default=None,
                           help="If set, use this as name of base model.")
    argparser.add_argument("--wandb_mode", type=str, default="online",
                           choices=["online", "offline", "disabled"],
                           help="WandB mode for logging. 'online' for normal logging, 'offline' for local runs, 'disabled' to turn off WandB logging.")
    argparser.add_argument("--metric", type=str, required=True, help="Metric to look for best model")
    args = argparser.parse_args()
    if not args.resume_training:
        args.RUN_NAME = f"{args.base_model.replace('/', '-')}_{args.trainer_type}_epochs_{args.num_epochs}_batch_{args.batch_size * args.gradient_accumulation}_lr_{args.lr}_seed_{args.seed}_preprocessed"
    else:
        args.RUN_NAME = f"{args.resume_training}_{args.trainer_type}_epochs_{args.num_epochs}_batch_{args.batch_size * args.gradient_accumulation}_lr_{args.lr}_seed_{args.seed}_preprocessed"
    if args.only_vision:
        args.RUN_NAME += "_only_vision"
    if args.frozen_layers:
        args.RUN_NAME += "_frozen_layers"
    if args.projection_layers:
        args.RUN_NAME += "_projections"
    if args.projection_vision:
        args.RUN_NAME += "_proj_vision"
    if args.projection_text:
        args.RUN_NAME += "_proj_text"
    if args.ds_size > 0:
        args.RUN_NAME += f"_ds_size_{args.ds_size}"
    if args.extra_name:
        args.RUN_NAME += f"_{args.extra_name}"
    print("Running a single training run.")
    args.project_name = "clip-multipositive-training"
    train_single_run(config=vars(args), fixed_args=args)


if __name__ == "__main__":
    main()
