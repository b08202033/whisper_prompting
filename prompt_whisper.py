import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import WhisperProcessor, WhisperTokenizer
import argparse
from opencc import OpenCC
from jiwer import wer
import json
from tqdm import tqdm
import re
from model import MyWhisperForConditionalGeneration
import pandas as pd
import whisper
import numpy as np
from transformers import GenerationConfig
import os
cc = OpenCC('t2s')

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, default="transcribe")
    parser.add_argument('--language', '-l', type=str)
    parser.add_argument('--prompt', '-p', type=str)
    parser.add_argument('--model_name_or_path', '-m', type=str, default="openai/whisper-large-v3")
    parser.add_argument('--output_dir', '-o', type=str, default="./test_output")
    parser.add_argument('--exp_name', '-n', type=str, default="results")
    parser.add_argument('--overwrite_forced_decoder_ids', '-c', type=str) # "<|zh|><|en|><|transcribe|><|notimestamps|>"
    parser.add_argument('--split', '-s', type=str, default="test")
    parser.add_argument('--label', '-b', type=int)
    parser.add_argument('--dataset_path', '-d', type=str, default='./zh_en')
    parser.add_argument('--example', '-e', type=str)
    parser.add_argument('--cache_dir', type=str, default="./cache")
    return parser.parse_args()

def insert_space_in_code_switched_text(text):
    text = text.lower()

    # Regular expression to match Chinese characters
    chinese_char_pattern = r'[\u4e00-\u9fff]'

    # Insert space before and after each Chinese character
    spaced_text = re.sub(f'({chinese_char_pattern})', r' \1 ', text)

    # Remove punctuations
    spaced_text = re.sub(r'[^\w\s]', '', spaced_text)

    # Remove any extra spaces added by the previous step
    normalized_text = re.sub(r'\s+', ' ', spaced_text)
    normalized_text = normalized_text.strip().replace("  ", " ")
    return normalized_text

def whisper_feature_extractor(raw_audio: np.array):
    audio_padded = whisper.pad_or_trim(raw_audio.flatten())
    input_feature = whisper.log_mel_spectrogram(audio_padded)
    return input_feature

# Calculate MER
def calculate_MER(results):
    hyps = []
    refs = []
    new_results = []
    for result in results:
        p = insert_space_in_code_switched_text(cc.convert(result["prediction"]))
        hyps.append(p)

        t = insert_space_in_code_switched_text(cc.convert(result["transcription"]))
        refs.append(t)

        new_results.append({
            "id": result["id"],
            "prediction": p,
            "transcription": t,
            "raw_prediction": result["prediction"],
        })

    return new_results, wer(refs, hyps)

def main(args):
    example = None
    
    if not os.path.exists(args.output_dir):
        print("Create output directory:", args.output_dir)
        os.makedirs(args.output_dir)
    else:
        print("Output directory exists:", args.output_dir)
        
    # Load dataset
    DATASET_PATH = args.dataset_path
    dataset = load_dataset(DATASET_PATH, split=args.split, cache_dir=args.cache_dir)

    if DATASET_PATH == "ky552/cszs_fr_en" or DATASET_PATH == "ky552/cszs_zh_en":
        dataset = dataset.remove_columns(["wrong_audio", 'wrong_transcription', 'wrong_file'])
        dataset = dataset.rename_column("correct_audio", "audio")
        dataset = dataset.rename_column("correct_transcription", "transcription")
        dataset = dataset.rename_column("correct_file", "file")

    # dataset = Dataset.from_dict(dataset) # DO NOT MODIFY
    print("="*15, "Dataset Info", "="*15)
    print("Dataset:", DATASET_PATH)

    if args.label is not None:
        dataset = dataset.filter(lambda x: x['label'] == args.label)
        assert dataset[0]['label'] == args.label
    
    # Load model
    print("="*15, "Model Info", "="*17)
    device = "cuda"
    model_name_or_path = args.model_name_or_path
    model = MyWhisperForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=args.cache_dir).to(device)
    
    # Some models don't have a preprocessor to load. We initialize from openai's
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, cache_dir=arg.cache_dir) 
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    processor.tokenizer = tokenizer
    print("Model: ", model_name_or_path)
    print("Tokenizer: ", processor.tokenizer.name_or_path)

    # Create dataloader
    def collate_fn(batch):
        file = []
        for item in batch:
            if 'label' in dataset.features.keys():
                label = 'correct' if item['label'] == 0 else 'wrong'
                file.append(f'{label}/' + item['audio']['path'])
            else:
                file.append(item['file'])

        if example is not None:
            audio = [np.hstack((example['audio']['array'], item['audio']["array"])) for item in batch]
        else:
            audio = [item['audio']["array"] for item in batch]
        transcription = [insert_space_in_code_switched_text(item['transcription']) for item in batch]
        inputs = {
            "file": file,
            "audio": processor(audio, sampling_rate=16000, return_tensors="pt").input_features,
            "transcription": transcription
        }

        return inputs
    
    batch_size = 32
    dataloader = DataLoader(
        dataset,  
        batch_size = batch_size,
        collate_fn = collate_fn,
    )

    # Start Inference
    model.eval()
    prompt = args.prompt
    prompt_ids = processor.get_prompt_ids(prompt) if prompt else None
    example_ids = processor.get_prompt_ids(example['transcription']) if example is not None else None

    if args.overwrite_forced_decoder_ids is not None:
        overwrite_forced_decoder_ids = []
        token_ids = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize(args.overwrite_forced_decoder_ids))
        for i, token_id in enumerate(token_ids):
            overwrite_forced_decoder_ids.append((i+1, token_id))
    else:
        overwrite_forced_decoder_ids = None

    generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    model.generation_config = generation_config
    generate_options = {
        "language": args.language,
        "prompt_ids": prompt_ids,
        "task": args.task,
        "overwrite_force_decoder_ids": overwrite_forced_decoder_ids,
        "example_ids": example_ids,
    }

    print("="*15, "Inference Info", "="*15)
    print("batch_size:", batch_size)
    print("generate_options:", generate_options)

    results = []
    for b in tqdm(dataloader):
        generated_ids, generation_config = model.generate(
            inputs=b["audio"].to(device),
            generation_config=generation_config,
            **generate_options
        )
        
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for file, t, p in zip(b["file"], b["transcription"], predictions):
            if prompt is not None:
                p = p.replace(" "+prompt, "", 1) # remove prompt

            if example is not None:
                p = p.replace(example['transcription'], "", 1) # remove example
                p = p.strip()
            
            results.append({
                "id": file,
                "prediction": p,
                "transcription": t
            })

        # break # for debug

    # print("forced_decoder_ids:", processor.decode([y for x,y in generation_config.forced_decoder_ids])) # for debug
    # assert False
    generate_options["prompt_ids"] = generate_options["prompt_ids"].tolist() if prompt is not None else None
    generate_options['prompt'] = prompt if prompt is not None else None
    generate_options["example_ids"] = generate_options["example_ids"].tolist() if example is not None else None
    generate_options["example"] = example['transcription'] if example is not None else None

    json.dump(
        {"model_name_or_path": model_name_or_path, "generate_options": generate_options,"raw_results": results},open(f"{args.output_dir}/{args.exp_name}.json", "w", encoding='utf16'), indent=2, ensure_ascii=False
    )

    results, word_error_rate = calculate_MER(results)
    print("Number of data:", len(results))
    print("WER:", word_error_rate)

    json.dump(
        {"model_name_or_path": model_name_or_path, "MER": word_error_rate, "generate_options": generate_options,"results": results},open(f"{args.output_dir}/{args.exp_name}.json", "w", encoding='utf16'), indent=2, ensure_ascii=False
    )

    print(f"Output file: {args.output_dir}/{args.exp_name}.json")


if __name__ == "__main__":
    args = get_args_parser()
    main(args=args)