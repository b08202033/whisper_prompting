# DO PROMPTS REALLY PROMPT? Exploring the Prompt Understanding Capability of Whisper

## Datasets
All the used datasets are publicly available. Here's the links:
- GigaSpeech: https://huggingface.co/datasets/speechcolab/gigaspeech
- ASCEND: https://huggingface.co/datasets/CAiRE/ASCEND
- CSZS-correct-zh: https://huggingface.co/datasets/ky552/cszs_zh_en
- CSZS-correct-fr: https://huggingface.co/datasets/ky552/cszs_fr_en

## Usage
Here are some examples of running our codes. Feel free to modify the arguments and the codes if needed.
* Code-switched ASR
Here's an example for running on CSZS-correct-zh and CSZS-correct-fr. 
```
python prompt_whisper.py -t transcribe -l zh -m "openai/whisper-large-v3" \\
-o test_output -n results -c "<|zh|><|en|><|transcribe|><|notimestamps|>" \\
-s test -d ky552/cszs_zh_en
```
And and example for running on ASCEND
```
python prompt_whisper_ASCEND.py -t transcribe -l zh -m "openai/whisper-large-v3" \\
-o test_output -n results -c "<|zh|><|en|><|transcribe|><|notimestamps|>" \\
-s test -d CAiRE/ASCEND
```

Particularly, you can change the provided language tokens through ``-c`` argument.

* Word count of a specific languages
Please change the ``languages`` and ``file_name`` parameters in ``count_language.py``, and run:
```
python count_language.py
```


* Textual Prompts
For the templates and keywords, please refer to ``templates.txt`` and ``keywords.json``

For the experiments of textual prompts on ASCEND-zh and ASCEND-en, you can run like:
```
python prompt_whisper_multidomain_ASCEND.py -t transcribe -l zh -m "openai/whisper-large-v3" \\
-o test_output -n results -s test -d CAiRE/ASCEND \\
--prompt_domain education --real_domain sports --subset_language zh
```
You can change the language of subset through ``--subset_language``, which is either "zh" or "en".
The arguments ``--prompt_domain`` and ``--real_domain`` represent the topics in prompt and the dataset, respectively.

For the experiments of textual prompts on GigaSpeech, you can run like:
```
python prompt_whisper_gigaspeech.py -t transcribe -l en -m "openai/whisper-large-v3" \\
-o test_output -n results -d speechcolab/gigaspeech \\
--real_domain Arts
```
Again, ``--real_domain`` means the topic of the tested subset. Running this code will run over all the prompts with all the topics.

To change the prompt template, please modify the line 172 of ``prompt_whisper_multidomain_ASCEND.py`` and line 122 of ``prompt_whisper_gigaspeech.py``.

For evaluation, please provide the directories containing your results in line 16 of ``calculate_performance.py``, and simply run it:
```
python calculate_performance.py
```

