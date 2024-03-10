import re

conversational_filler = ['UH', 'UHH', 'UM', 'EH', 'MM', 'HM', 'AH', 'HUH', 'HA', 'ER', 'OOF', 'HEE' , 'ACH', 'EEE', 'EW']
unk_tags = ['<UNK>', '<unk>']
gigaspeech_punctuations = ['<COMMA>', '<PERIOD>', '<QUESTIONMARK>', '<EXCLAMATIONPOINT>']
gigaspeech_garbage_utterance_tags = ['<SIL>', '<NOISE>', '<MUSIC>', '<OTHER>']
non_scoring_words = gigaspeech_punctuations + gigaspeech_garbage_utterance_tags + conversational_filler + unk_tags

def gigaspeech_process(text):
    # remove hyphen
    text = text.replace('-', ' ')

    # remove unscoring words
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)
    
    text = ' '.join(remaining_words)

    # lowercase
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().replace("  ", " ")

    return text