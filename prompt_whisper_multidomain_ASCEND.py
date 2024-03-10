# This is code for prompting whisper on CS Zerospeech and ML2021 Corpus

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
# import customize code
from model import MyWhisperForConditionalGeneration
import pandas as pd
import whisper
import numpy as np
from transformers import GenerationConfig
import os
from prompt_whisper import insert_space_in_code_switched_text
cc = OpenCC('t2s')

simplied_chinese_topic = {
    "education": "教育",
    "persona": "人格",
    "technology": "科技",
    "philosophy": "哲学",
    "sports": "运动"
}

keyword_list = {
    "education": ["Education", "learning", "teaching", "university"],
    "persona": ["Persona", "character", "identity", "hobby"],
    "technology": ["Technology", "social media", "mobile applications", "smart devices"],
    "philosophy": ["Philosophy", "ethics", "dilemma", "morality"],
    "sports": ["Sports", "exercise", "game", "fitness"],
}

chinese_keyword_list = {
    "education": ["教育", "学习", "教学", "大学"],
    "persona": ["人格", "個性", "身份", "爱好"],
    "technology": ["科技", "社交媒体", "移动应用程序", "智能设备"],
    "philosophy": ["哲学", "伦理", "困境", "道德"],
    "sports": ["运动", "锻炼", "游戏", "健身"],
}

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, default="transcribe")
    parser.add_argument('--language_tag', '-l', type=str)
    parser.add_argument('--prompt', '-p', type=str)
    parser.add_argument('--model_name_or_path', '-m', type=str, default="openai/whisper-large-v3")
    parser.add_argument('--output_dir', '-o', type=str, default="./test_output")
    parser.add_argument('--exp_name', '-n', type=str, default="results")
    parser.add_argument('--overwrite_forced_decoder_ids', '-c', type=str)
    parser.add_argument('--split', '-s', type=str, default="test")
    parser.add_argument('--dataset_path', '-d', type=str, default='CAiRE/ASCEND')
    parser.add_argument('--example', '-e', type=str) # format: train/correct/0.wav
    parser.add_argument('--use_domain_tag', '-u', action='store_true')
    parser.add_argument('--prompt_domain', type=str)
    parser.add_argument('--real_domain', type=str, default="education")
    parser.add_argument('--subset_language', type=str)
    parser.add_argument('--cache_dir', type=str, default="./cache")
    return parser.parse_args()

def whisper_feature_extractor(raw_audio: np.array):
    audio_padded = whisper.pad_or_trim(raw_audio.flatten())
    input_feature = whisper.log_mel_spectrogram(audio_padded)
    return input_feature

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

    # spliting topics
    topics = ['education', 'persona', 'technology', 'philosophy', 'sports']
    languages = ['zh', 'en', 'mixed']
    assert args.real_domain in topics, "The real domain is not in the list of topics"

    if args.prompt_domain is not None:
        assert args.prompt_domain in topics, "The prompt domain is not in the list of topics"

    if args.subset_language is not None:
        assert args.subset_language in languages, "The subset language is not in the list of languages"
    # topic2dataset = {}
    # total_size = 0
    # for topic in topics:
    #     topic_dataset = dataset.filter(lambda x: x['topic'] == topic)
    #     print(f"{topic} dataset size: {topic_dataset.num_rows}")

    #     if topic_dataset.num_rows > 0:
    #         topic2dataset[topic] = topic_dataset
    #         total_size += topic2dataset[topic].num_rows
    
    # assert total_size == dataset.num_rows, "The sum of all topics is not equal to the total size of the dataset. Some topics are missing."
    topic_dataset = dataset.filter(lambda x: (x['topic'] == args.real_domain and x['language']==args.subset_language) if args.subset_language is not None else x['topic'] == args.real_domain)


    
    # dataset = Dataset.from_dict(dataset) # DO NOT MODIFY
    print("="*15, "Dataset Info", "="*15)
    print("Dataset:", DATASET_PATH)
    
    # Load model
    print("="*15, "Model Info", "="*15)
    device = "cuda"
    model_name_or_path = args.model_name_or_path
    model = MyWhisperForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=args.cache_dir).to(device)
    
    # Some models don't have a preprocessor to load. We initialize from openai's
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, cache_dir=args.cache_dir) 
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    processor.tokenizer = tokenizer
    print("Model: ", model_name_or_path)
    print("Tokenizer: ", processor.tokenizer.name_or_path)

    # Create dataloader
    def collate_fn(batch):
        file = []
        for item in batch:
            file.append(item['audio']['path'])

        if example is not None:
            audio = [np.hstack((example['audio']['array'], item['audio']["array"])) for item in batch]
            # audio = [item['audio']["array"] for item in batch]
        else:
            audio = [item['audio']["array"] for item in batch]
        transcription = [insert_space_in_code_switched_text(item['transcription']) for item in batch]
        inputs = {
            "file": file,
            "audio": processor(audio, sampling_rate=16000, return_tensors="pt").input_features,
            "transcription": transcription
        }

        return inputs

    # Start Inference
    model.eval()

    if args.overwrite_forced_decoder_ids is not None:
        overwrite_forced_decoder_ids = []
        token_ids = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize(args.overwrite_forced_decoder_ids))
        for i, token_id in enumerate(token_ids):
            overwrite_forced_decoder_ids.append((i+1, token_id))
    else:
        overwrite_forced_decoder_ids = None

    results = []

    batch_size = 32
    dataloader = DataLoader(
        topic_dataset,  
        batch_size = batch_size,
        collate_fn = collate_fn,
    )

    generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    model.generation_config = generation_config

    assert (args.prompt is None) or (args.use_domain_tag is False), "Custom prompts and domain tags cannot be used at the same time."
    
    if args.prompt_domain is not None:
        domain_prompt = f'So we were just talking about {args.prompt_domain}'
        prompt = args.prompt + ' ' + domain_prompt if args.prompt is not None else domain_prompt
    
    else:
        prompt = args.prompt

    print("Prompt:", prompt)
    
    prompt_ids = processor.get_prompt_ids(prompt) if prompt else None
    
    
    example_ids = processor.get_prompt_ids(example['transcription']) if example is not None else None
    generate_options = {
        "language": args.language_tag,
        "prompt_ids": prompt_ids,
        "task": args.task,
        "overwrite_force_decoder_ids": overwrite_forced_decoder_ids,
        "example_ids": example_ids,
    }


    print("="*15, "Inference Info", "="*15)
    print("batch_size:", batch_size)
    print("generate_options:", generate_options)

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

            results.append({
                "id": file,
                "prediction": p,
                "transcription": t,
            })
        

    print("forced_decoder_ids:", processor.decode([y for x,y in generation_config.forced_decoder_ids])) # for debug

    # Calculate MER
    def calculate_MER(results):
        hyps = []
        refs = []
        new_results = []
        for result in results:
            p = cc.convert(result["prediction"])
            if example is not None:
                p = p.replace(example['transcription'], "", 1) # remove example
                p = p.strip()

            p = insert_space_in_code_switched_text(p)
            hyps.append(p)

            refs.append(cc.convert(result["transcription"]))

            new_results.append({
                "id": result["id"],
                "prediction": p,
                "transcription": result["transcription"],
                "raw_prediction": result["prediction"],
            })

        return new_results, wer(refs, hyps)

    results, word_error_rate = calculate_MER(results)
    print("Number of data:", len(results))
    print("WER:", word_error_rate)
    # assert False
    generate_options["prompt_ids"] = generate_options["prompt_ids"].tolist() if prompt is not None else None
    generate_options["example_ids"] = generate_options["example_ids"].tolist() if example is not None else None
    generate_options["example"] = example['transcription'] if example is not None else None

    output_file = f"{args.output_dir}/{args.exp_name}.json"

    json.dump(
        {"model_name_or_path": model_name_or_path,
         "language_tag": args.language_tag, 
         'subset_language': args.subset_language, 
         'real_domain': args.real_domain, 
         'prompt_domain': args.prompt_domain,
         'prompt': prompt, 
         "MER": word_error_rate, 
         "generate_options": generate_options,
         "results": results},
         
         open(output_file, "w", encoding='utf16'), indent=2, ensure_ascii=False
    )
    print(f"Output file: {output_file} is saved.")


if __name__ == "__main__":
    args = get_args_parser()
    main(args=args)