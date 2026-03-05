

import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


base_dir = Path('ms_asl')
train_json = base_dir/'MSASL_train_clean.json'

with open(train_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

labels = [i['clean_text'].lower() for i in data] # clean_text anahtarı üzerinden classları ve örnekleri sayar..!
counter = Counter(labels)

print(f'örnek sayısı -> {len(labels)}')
print(f'class sayısı -> {len(counter)}')

more_classes = counter.most_common(10) # çok fazla örneğe sahip 10 class
less_classes = counter.most_common()[-10:] # daha az örneğe sahip 10 class

print(f'çok fazla örneğe sahip 10 class')
for k,v in more_classes:
    print(f'{k:20s}: {v}')

print(f'daha az örneğe sahip 10 class')
for k,v in less_classes:
    print(f'{k:20s}: {v}')


counts = list(counter.values())
plt.figure(figsize=(12, 7))
plt.hist(counts, bins=50)
plt.yscale('log')
plt.xlabel('class örnek sayısı')
plt.ylabel('class sayısı (log scale)')
plt.title('MS-ASL Train Class Distribution')
plt.savefig('class_distribution.png')

threshold = 10
low_class = [k for k, v in counter.items() if v < threshold]
print(f'{threshold} az örneği olan class sayısı: {len(low_class)}')