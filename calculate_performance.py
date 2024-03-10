from prompt_whisper_gigaspeech import calculate_MER
import json
from jiwer import wer
from prompt_whisper import insert_space_in_code_switched_text
from gigaspeech_process import gigaspeech_process


lang = "en"
corpus = 'ascend'

if corpus == 'gigaspeech':
    domains = ["Arts", "Science and Technology", "Nonprofits and Activism", "Sports"]
else:
    domains = ['education', 'persona', 'technology', 'philosophy', 'sports']

# ASCEND-en english prompt
folder_names= [
    "multidomain_ASCEND/whisper_transcribe_video_subset_en",
    "multidomain_ASCEND/whisper_talk_about_subset_en",
    "multidomain_ASCEND/whisper_keyword_list_subset_en",
    "multidomain_ASCEND/whisper_topic_of_this_talk_subset_en",
    "multidomain_ASCEND/whisper_describe_exp_subset_en",
    "multidomain_ASCEND/subset_en",
    "multidomain_ASCEND/whisper_clair_subset_en",
    "multidomain_ASCEND/subset_en_single_word_prompt",
    "multidomain_ASCEND/whisper_think_about_subset_en",
    "multidomain_ASCEND/whisper_share_something_subset_en"
]

folder_map = {
    "Arts": "arts",
    "Science and Technology": "science",
    "Nonprofits and Activism": "nonprofits",
    "Sports": "sports"
}

TFRs = []
PERFs = []
BPERFs = []
for f in folder_names:
    # TFR
    folder_name = f.split("/")[-1]
    count = 0
    for d in domains:
        s = '_'.join(d.split())
        min_wer = 1000
        min_domain = None
        source_wer = 0
        for j in domains:
            p = '_'.join(j.split())
            if corpus == 'gigaspeech':
                file_name = f"./multidomain_gigaspeech/{folder_name}_source_{'_'.join(folder_map[d].split())}/real_{s}_prompt_{p}.json"
            else:
                file_name = f"./multidomain_ASCEND/{folder_name}/subset_{lang}_real_{s}_prompt_{p}.json"
            with open(file_name, encoding='utf-16') as f:
                data = json.load(f)
            if j==d:
                source_wer = data['MER']
            if data['MER'] <= min_wer:
                min_wer = data['MER']
                min_domain = j
        
        count += int(d==min_domain or source_wer==min_wer)

    print("#"*100)
    print(f"TFR of {folder_name}: {count / len(domains) *100}%")
    TFRs.append(count / len(domains) *100)
    # PERF
    results = []
    for d in domains:
        s = '_'.join(d.split())
        
        if corpus == 'gigaspeech':
            file_name = f"./multidomain_gigaspeech/{folder_name}_source_{'_'.join(folder_map[d].split())}/real_{s}_prompt_{s}.json"
        else:
            file_name = f"./multidomain_ASCEND/{folder_name}/subset_{lang}_real_{d}_prompt_{d}.json"
        with open(file_name, encoding='utf-16') as f:
            data = json.load(f)
        results.extend(data['results'])

    if corpus == 'gigaspeech':
        new_results, performance = calculate_MER(results)
    else:
        performance = wer([r['transcription'] for r in results], [r['prediction'] for r in results])
    print(f"Performance of {folder_name}: {performance*100} %")
    PERFs.append(performance*100)
    # BPERF
    results2 = []
    for i in domains:
        min_wer = 1000
        extension = []
        for j in domains:
            a = '_'.join(i.split())
            b = '_'.join(j.split())
            
            if corpus == 'gigaspeech':
                file_name = f"./multidomain_gigaspeech/{folder_name}_source_{folder_map[i]}/real_{a}_prompt_{b}.json"
            else:
                file_name = f"./multidomain_ASCEND/{folder_name}/subset_{lang}_real_{i}_prompt_{j}.json"
            with open(file_name, encoding='utf-16') as f:
                data = json.load(f)
            if data['MER'] <= min_wer:
                min_wer = data['MER']
                extension = data['results']
        results2.extend(extension)
    if corpus == 'gigaspeech':
        new_results2, performance2 = calculate_MER(results2)
    else:
        performance2 = wer([r['transcription'] for r in results2], [r['prediction'] for r in results2])

    assert performance2 <= performance
    print(f"Best Performance of {folder_name}: {performance2*100} %")
    BPERFs.append(performance2*100)
    print("#"*100)

print(f"Average TFR: {sum(TFRs) / len(TFRs)}")
print(f"Average PERF: {sum(PERFs) / len(PERFs)}")
print(f"Average BPERF: {sum(BPERFs) / len(BPERFs)}")
