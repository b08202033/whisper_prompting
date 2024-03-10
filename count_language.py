import json
from lingua import Language, LanguageDetectorBuilder

languages = [Language.ENGLISH, Language.FRENCH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

file_name = "test.json"
with open(file_name, "r", encoding='utf-16') as f:
    data = json.load(f)

results = []

count_pred = 0
count_ref = 0
for i in range(len(data['results'])):
    if 'correct' in data['results'][i]['id']:
        pred = data['results'][i]['prediction']
        ref = data['results'][i]['transcription']

        preds = pred.split(' ')
        for w in preds:
            count_pred += detector.detect_language_of(w) == Language.FRENCH
        refs = ref.split(' ')
        for w in refs:
            count_ref += detector.detect_language_of(w) == Language.FRENCH

print(f'{count_pred} / {count_ref} = {count_pred/count_ref*100}%')