

import json
from collections import Counter
import os


base_dir = 'ms_asl'
threshold = 15

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        return json.dump(data, f, indent=2)


def get_class_counts(train_data):
    labels = []
    for i in train_data:
        labels.append(i['label'])
    return Counter(labels)


def filter_split(data, valid_labels):
    filter_data = []
    for i in data:
        if i['label'] in valid_labels:
            filter_data.append(i)
    return filter_data


def label_remapping(data, label_map): # label'ları 0'dan başlayacak şekilde ardışık olarak yeniden numaralandırır..!
    for i in data:
        i['label'] = label_map[i['label']]
    return data


def main():
    train_path = os.path.join(base_dir, 'MSASL_train_clean.json')
    val_path = os.path.join(base_dir, 'MSASL_val_clean.json')
    test_path = os.path.join(base_dir, 'MSASL_test_clean.json')

    train_data = load_json(train_path)
    val_data = load_json(val_path)
    test_data = load_json(test_path)

    class_counts = get_class_counts(train_data) # train'e göre valid label belirleme..!
    valid_labels = {label for label, cnt in class_counts.items() if cnt >= threshold}

    print(f'Toplam class -> {len(class_counts)}')
    print(f'Kalan class >= {threshold} -> {len(valid_labels)}')

    label_map = {old: new for new, old in enumerate(sorted(valid_labels))} # label mapping ile eski labellar yerine yeni labellar sırayla oluşur..!

    train_f = filter_split(train_data, valid_labels)
    val_f = filter_split(val_data, valid_labels)
    test_f = filter_split(test_data, valid_labels)

    label_remapping(train_f, label_map)
    label_remapping(val_f, label_map)
    label_remapping(test_f, label_map)

    save_json(train_f, os.path.join(base_dir, 'MSASL_train_filtered.json'))
    save_json(val_f, os.path.join(base_dir, 'MSASL_val_filtered.json'))
    save_json(test_f, os.path.join(base_dir, 'MSASL_test_filtered.json'))

    save_json(label_map, os.path.join(base_dir, 'label_remapping.json')) # label remapping'i de kaydettim..!

    print('\ntrain:')
    print(f'original: {len(train_data):> 6} -> filtered: {len(train_f)}')
    print('\nval:')
    print(f'original: {len(val_data):> 6} -> filtered: {len(val_f)}')
    print('\ntest:')
    print(f'original: {len(test_data):> 6} -> filtered: {len(test_f)}')

    print(f'\nLabel range kontrolü:')
    print(f'train min/max label: {min(i["label"] for i in train_f)} / {max(i["label"] for i in train_f)}')
    print(f'val min/max label: {min(i["label"] for i in val_f)} / {max(i["label"] for i in val_f)}')
    print(f'test min/max label: {min(i["label"] for i in test_f)} / {max(i["label"] for i in test_f)}')


if __name__ == '__main__':
    main()