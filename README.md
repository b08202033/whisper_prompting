# Do Prompts Really Prompt? Exploring the Prompt Understanding Capability of Whisper
<p align="center">
  <a href="https://2024.ieeeslt.org/">[SLT2024]</a> <a href="https://arxiv.org/abs/2406.05806">[arXiv]</a>
</p>

The official Github repository for the implementation of SLT2024 paper "Do Prompts Really Prompt? Exploring the Prompt Understanding Capability of Whisper".

- Authors: Chih-Kai Yang, Kuan-Po Huang, Hung-yi Lee  
- Affiliation: National Taiwan University

## Abstract
This research explores how the information of prompts interacts with the high-performing speech recognition model, Whisper. We compare its performances when prompted by prompts with correct information and those corrupted with incorrect information. Our results unexpectedly show that Whisper may not understand the textual prompts in a human-expected way. Additionally, we find that performance improvement is not guaranteed even with stronger adherence to the topic information in textual prompts. It is also noted that English prompts generally outperform Mandarin ones on datasets of both languages, likely due to differences in training data distributions for these languages despite the mismatch with pre-training scenarios. Conversely, we discover that Whisper exhibits awareness of misleading information in language tokens by ignoring incorrect language tokens and focusing on the correct ones. In sum, We raise insightful questions about Whisper's prompt understanding and reveal its counter-intuitive behaviors. We encourage further studies.

## Datasets
All the used datasets are publicly available. Here's the links:
- GigaSpeech: https://huggingface.co/datasets/speechcolab/gigaspeech
- ASCEND: https://huggingface.co/datasets/CAiRE/ASCEND
- CSZS-correct-zh: https://huggingface.co/datasets/ky552/cszs_zh_en
- CSZS-correct-fr: https://huggingface.co/datasets/ky552/cszs_fr_en  

As for the prompt templates, you can find them in ``templates.txt``

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

## Citation
If you find this paper interesting and useful, please consider citing the following papers:
```
@article{yang2024prompts,
  title={Do Prompts Really Prompt? Exploring the Prompt Understanding Capability of Whisper},
  author={Yang, Chih-Kai and Huang, Kuan-Po and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2406.05806},
  year={2024}
}

@INPROCEEDINGS{10446737,
  author={Huang, Kuan-Po and Yang, Chih-Kai and Fu, Yu-Kuan and Dunbar, Ewan and Lee, Hung-Yi},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Zero Resource Code-Switched Speech Benchmark Using Speech Utterance Pairs for Multiple Spoken Languages}, 
  year={2024},
  volume={},
  number={},
  pages={10006-10010},
  keywords={Speech coding;Benchmark testing;Signal processing;Linguistics;Acoustics;Speech processing;Task analysis;Code-switch;Multilingual;Discrete unit;Zero resource;Self-supervised},
  doi={10.1109/ICASSP48485.2024.10446737}}

@INPROCEEDINGS{yang2023investigating,
  author={Yang, Chih-Kai and Huang, Kuan-Po and Lu, Ke-Han and Kuan, Chun-Yi and Hsiao, Chi-Yuan and Lee, Hung-Yi},
  booktitle={2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)}, 
  title={Investigating Zero-Shot Generalizability on Mandarin-English Code-Switched ASR And Speech-to-Text Translation of Recent Foundation Models with Self-Supervision and Weak Supervision}, 
  year={2024},
  volume={},
  number={},
  pages={540-544},
  keywords={Speech coding;Conferences;Signal processing;Acoustics;Task analysis;Speech processing;Speech to text;Code-switch;Prompt;Speech recognition;Speech translation;Self-supervised},
  doi={10.1109/ICASSPW62465.2024.10626762}}
```

