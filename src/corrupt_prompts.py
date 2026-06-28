from vllm import LLM, SamplingParams
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import torch

ERROR_TYPES = [
    "main subject",
    "color of the main subject",
    "quantity of the main subject",
    "background details",
]

# Define the SYSTEM and USER prompts separately
SYSTEM_PROMPT = """You are a highly specialized text transformation AI. Your sole task is to receive an original sentence and generate a single, corrupted version of that sentence.

Rules for Corruption:
1.  **Maintain Original Structure:**  The overall sentence flow must remain identical to the original.
2.  **Targeted Non-Factuality:** Only modify the aspect of the sentence corresponding to the '{ERROR_TYPE}' provided. This modification should render that specific aspect non-factual or illogical within the context of the original sentence.
3.  **Single Output:** Output only the corrupted sentence. No additional text, explanations, or formatting.
4. **Coherent Output:** Ensure that the output is coherent with the new changes."""

USER_PROMPT_TEMPLATE = """Change the {ERROR_TYPE} in the following PROMPT:
PROMPT: {PROMPT}"""


def create_prompts(ds):
    # This will now store lists of dictionaries, suitable for chat_template
    prompts_for_llm_generate = []
    # This will store the original 'caption_llava_short' for dataset creation
    original_captions_key = []

    for item in tqdm(ds, desc="Creating prompts"):
        prompt_text = item['caption_llava_short']
        original_captions_key.append([item['key'], prompt_text])
        for error_type in ERROR_TYPES:
            # Construct the list of messages for the current conversation
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(ERROR_TYPE=error_type)},
                # System message includes error_type for completeness, though it's not strictly necessary for rules
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(ERROR_TYPE=error_type, PROMPT=prompt_text)}
            ]
            # Append this list of messages (conversation history)
            prompts_for_llm_generate.append(messages)

    return prompts_for_llm_generate, original_captions_key


def generate_non_factual_prompts(model_name, chat_prompts):
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
    llm = LLM(model=model_name,
              dtype=torch.bfloat16,
              trust_remote_code=True,
              quantization="bitsandbytes",
              limit_mm_per_prompt={"image": 0},
              max_model_len=16384)

    outputs = llm.chat(chat_prompts, sampling_params)
    return outputs


def create_dataset_from_outputs(original_caption_keys, outputs, output_path):
    new_ds_rows = []

    current_original_caption_index = 0
    generated_texts_for_current_caption = []

    for i, output in enumerate(tqdm(outputs, desc="Processing outputs for dataset creation")):
        generated_text = output.outputs[0].text
        generated_texts_for_current_caption.append(generated_text)

        if (i + 1) % len(ERROR_TYPES) == 0:
            temp_dict = {
                "key": original_caption_keys[current_original_caption_index][0],
                # Use the key from the original captions
                "caption_llava_short": original_caption_keys[current_original_caption_index][1],
                # Use the caption from the original captions
            }

            for error_type_index, error_type in enumerate(ERROR_TYPES):
                temp_dict[f"generated_text_{error_type.replace(" ", "_")}"] = generated_texts_for_current_caption[error_type_index]

            new_ds_rows.append(temp_dict)

            generated_texts_for_current_caption = []
            current_original_caption_index += 1

    # In case there's a partial batch at the end (shouldn't happen if data is clean)
    if generated_texts_for_current_caption:
        temp_dict = {
            "key": original_caption_keys[current_original_caption_index][0],
            "caption_llava_short": original_caption_keys[current_original_caption_index][1],
        }

        for error_type_index, error_type in enumerate(ERROR_TYPES):
            temp_dict[f"generated_text_{error_type}"] = generated_texts_for_current_caption[error_type_index]

        new_ds_rows.append(temp_dict)

    new_dataset = Dataset.from_list(new_ds_rows)
    new_dataset.save_to_disk(output_path)


if __name__ == '__main__':
    DATASET_PATH = "FILL"
    MODEL_NAME = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
    OUTPUT_PATH = "FILL"

    print(f"Loading dataset from {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    print(dataset)

    print("Creating prompts...")
    # create_prompts now returns two lists
    chat_prompts_for_llm, original_captions_keys = create_prompts(dataset)

    print("\nExample of a chat-formatted prompt (first one):")
    print(chat_prompts_for_llm[0])
    print(original_captions_keys[0])
    exit()
    print("\nGenerating non-factual prompts...")
    outputs = generate_non_factual_prompts(MODEL_NAME, chat_prompts_for_llm)

    print("\nGenerated Outputs (first few examples):")
    for i, output in enumerate(tqdm(outputs, desc="Processing outputs")):
        generated_text = output.outputs[0].text
        print(f"Output {i}: {generated_text}")
        if i >= 4:  # Just print a few examples
            break

    print("\nCreating dataset from outputs...")
    create_dataset_from_outputs(original_captions_keys, outputs, OUTPUT_PATH)

    print(f"\nDataset saved to {OUTPUT_PATH}")
    # Load and print a sample of the new dataset to verify
    new_ds_sample = load_from_disk(OUTPUT_PATH)
    print("\nSample of the created dataset:")
    print(new_ds_sample[0])  # Print the first row
