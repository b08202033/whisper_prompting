# This is code for prompting whisper on CS Zerospeech and ML2021 Corpus

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import WhisperProcessor, WhisperTokenizer
import argparse
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
from gigaspeech_process import gigaspeech_process
from opencc import OpenCC
cc = OpenCC('t2s')

categories = ["People and Blogs", "Business", "Nonprofits and Activism", "Crime", "History", "Pets and Animals", 
              "News and Politics", "Travel and Events", "Kids and Family", "Leisure", "N/A", "Comedy", "News and Politics", 
              "Sports", "Arts", "Science and Technology", "Autos and Vehicles", "Science and Technology", "People and Blogs", 
              "Music", "Society and Culture", "Education", "Howto and Style", "Film and Animation", "Gaming", "Entertainment", 
              "Travel and Events", "Health and Fitness", "audiobook"]

keyword_list = {
    "Science and Technology": ["Science and technology", "web", 'innovation', "media"],
    "Sports": ["Sports", "exercise", "game", "fitness"],
    "Arts": ["Arts", "culture", "performing", "visual"],
    "Nonprofits and Activism": ["Nonprofits and activism", "society", "organization", "social justice"],
}

random_prompts = ["Gardening, Culinary, Travel, Fashion", 
                  "Home Decor, Parenting, Pets, Fitness", 
                  "Health and Wellness, Outdoor Activities, Do It Yourself, Finance and Investing",
                  "Photography, Knitting, Beauty and Skincare, Paranormal"]

chinese_topic = {
    "Arts": "艺术",
    "Science and Technology": "科學與科技",
    "Nonprofits and Activism": "非營利組織與社會運動",
    "Sports": "運動"
}

# Calculate MER
def calculate_MER(results):
    hyps = []
    refs = []
    new_results = []
    for result in results:
        # if example is not None:
        #     p = p.replace(example['transcription'], "", 1) # remove example
        #     p = p.strip()
        if len(result['transcription']) == 0:
            continue
        p = result["prediction"]
        #p = gigaspeech_process(p.upper())
        #p = insert_space_in_code_switched_text(p)
        hyps.append(p)

        refs.append(result["transcription"])

        new_results.append({
            "id": result["id"],
            "prediction": p,
            "transcription": result["transcription"],
            "raw_prediction": result["prediction"],
        })

    return new_results, wer(refs, hyps)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, default="transcribe")
    parser.add_argument('--language_tag', '-l', type=str, default="en")
    parser.add_argument('--prompt', '-p', type=str)
    parser.add_argument('--model_name_or_path', '-m', type=str, default="openai/whisper-large-v3")
    parser.add_argument('--output_dir', '-o', type=str, default="./test_output")
    parser.add_argument('--exp_name', '-n', type=str, default="results")
    parser.add_argument('--overwrite_forced_decoder_ids', '-c', type=str)
    parser.add_argument('--split', '-s', type=str, default="train")
    parser.add_argument('--dataset_path', '-d', type=str, default="speechcolab/gigaspeech")
    parser.add_argument('--example', '-e', type=str) # format: train/correct/0.wav
    parser.add_argument('--use_domain_tag', '-u', action='store_true')
    # parser.add_argument('--prompt_domain', type=str, default="People and Blogs")
    parser.add_argument('--real_domain', type=str, default="Arts")
    parser.add_argument('--cache_dir', type=str, default="./cache/gigaspeech_s")
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def whisper_feature_extractor(raw_audio: np.array):
    audio_padded = whisper.pad_or_trim(raw_audio.flatten())
    input_feature = whisper.log_mel_spectrogram(audio_padded)
    return input_feature

def main(args):
    example = None
    if args.example is not None:
        print("-" * 100)
        print("In-context learning is activated. Checking the format.") # Currently only support one example for all the dataset

        example_dataset = load_dataset(args.dataset_path, cache_dir="./cache")
        tmp_dataset = None

        # case 1: ML2021_ASR
        if 'label' not in example_dataset['test'].features.keys():
            for split in example_dataset.keys():
                tmp_dataset = example_dataset[split].filter(lambda x: args.example in x['audio']['path'])
                if tmp_dataset.num_rows == 1:
                    break
            assert tmp_dataset.num_rows == 1, "The example path is not unique in the dataset"

            example = tmp_dataset[0]
            example['transcription'] = example['transcription'] + '。'
            del tmp_dataset
            del example_dataset

        # case 2: CS Zerospeech
        else:
            example_list = args.example.split("/")
            assert (example_list[0] == 'train') or (example_list[0] == 'test') or (example_list[0] == 'dev'), "The split should be either train, test, or dev"
            assert (example_list[1] == 'correct') or (example_list[1] == 'wrong'), "The label should be either correct or wrong"
            assert '.wav' in example_list[2], "The audio file should be in .wav format"
            
            print("Finding the example in the dataset.")
            dataset = load_dataset(args.dataset_path, split=example_list[0], cache_dir="./cache")
            dataset = dataset.filter(lambda x: x['label'] == 0 if example_list[1] == 'correct' else x['label'] == 1)
            dataset = dataset.filter(lambda x: x['audio']['path'] == example_list[2])
            assert dataset.num_rows > 0, "The example is not found in the dataset"
            assert dataset.num_rows == 1, "The example is not unique in the dataset"
            print("The example is found in the dataset.")
            
            example = dataset[0]
            if example['transcription'][-1] == '.':
                example['transcription'] = example['transcription'][:-1] + '。'
            elif example['transcription'][-1] != '。':
                example['transcription'] = example['transcription'] + '。'
                
            del dataset
        
        assert example is not None, "The example is not found in the dataset"
        print("Example transcription:", example['transcription'])

    if not os.path.exists(args.output_dir):
        print("Create output directory:", args.output_dir)
        os.makedirs(args.output_dir)
    else:
        print("Output directory exists:", args.output_dir)
        
    # Load dataset
    DATASET_PATH = args.dataset_path
    dataset = load_dataset(DATASET_PATH, "s", use_auth_token=True, cache_dir=args.cache_dir)
    dataset = dataset[args.split].remove_columns(["segment_id", 'speaker', 'begin_time', 'end_time', 'audio_id', 'title', 'url', 'original_full_path'])
    if args.debug: 
        dataset = dataset.select(range(1000))
        print(dataset)

    assert args.real_domain in categories, "The real domain is not in the list of categories"
    
    prompt_domain = ["Arts", "Science and Technology", "Nonprofits and Activism", "Sports"]
    
    prompts = [f"Let me share something about {domain.lower()} with you." for domain in prompt_domain]
    print(prompts)
    topic_dataset = dataset.filter(lambda x: categories[x['category']] == args.real_domain)
    assert topic_dataset.num_rows > 0, "The real domain is not found in the dataset"

    # if args.debug:
    #     topic_dataset = topic_dataset[:64]

    

    print("="*15, "Dataset Info", "="*15)
    print("Dataset:", DATASET_PATH)
    
    # Load model
    print("="*15, "Model Info", "="*15)
    device = "cuda"
    model_name_or_path = args.model_name_or_path
    model = MyWhisperForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir="./cache").to(device)
    
    # Some models don't have a preprocessor to load. We initialize from openai's
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, cache_dir="./cache") 
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
        transcription = [gigaspeech_process(item['text']) for item in batch]
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

    

    batch_size = 32
    dataloader = DataLoader(
        topic_dataset,  
        batch_size = batch_size,
        collate_fn = collate_fn,
    )


    generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    model.generation_config = generation_config

    assert (args.prompt is None) or (args.use_domain_tag is False), "Custom prompts and domain tags cannot be used at the same time."
    # domain_prompt = f'This utterance is about {args.prompt_domain}.'

    prompts = [args.prompt + ' ' + prompt for prompt in prompts] if args.prompt is not None else prompts
    for i in range(len(prompts)):
        results = []
        prompt = prompts[i]
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

            # break # for debug
        # print(generation_config.forced_decoder_ids)
        print("forced_decoder_ids:", processor.decode([y for x,y in generation_config.forced_decoder_ids])) # for debug

        

        results, word_error_rate = calculate_MER(results)
        print("Number of data:", len(results))
        print("WER:", word_error_rate)
        # assert False
        generate_options["prompt_ids"] = generate_options["prompt_ids"].tolist() if prompt is not None else None
        # generate_options['prompts'] = prompts if len(prompts) > 0 else None
        generate_options["example_ids"] = generate_options["example_ids"].tolist() if example is not None else None
        generate_options["example"] = example['transcription'] if example is not None else None

        output_file = f"{args.output_dir}/real_{'_'.join(args.real_domain.split())}_prompt_{'_'.join(prompt_domain[i].split())}.json"
        json.dump(
            {"model_name_or_path": model_name_or_path,
            "language_tag": args.language_tag, 
            'real_domain': args.real_domain, 
            'prompt_domain': prompt_domain[i],
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