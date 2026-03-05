

import json
from pathlib import Path
from tqdm import tqdm

base_dir = Path('ms_asl')

json_paths = {
    'train': base_dir/'MSASL_train.json',
    'val': base_dir/'MSASL_val.json',
    'test': base_dir/'MSASL_test.json'
}

video_dirs = {
    'train': base_dir/'videos'/'train'/'MS-ASL-ALL',
    'val': base_dir/'videos'/'val'/'MS-ASL-ALL',
    'test': base_dir/'videos'/'test'/'MS-ASL-ALL'
}


def clean_split(split): # videolarla json dosyalarını eşleştirip temizliyorum..!
    json_path = json_paths[split]
    video_dir = video_dirs[split]
    classes_path = base_dir/'MSASL_classes.json'

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(classes_path, 'r', encoding='utf-8') as f:
        words = json.load(f)

    cleaned = []
    missing = 0

    for idx, item in enumerate(tqdm(data, desc=f'check {split}')):
        label = item['label']
        title = words[label]
        item['vid_index'] = idx

        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        video_name = f'{safe_title}_{idx}.mp4' # diskteki dosya adı {safe_title}_{index}.mp4 olarak kayıtlı..!
        video_path = video_dir/video_name

        if video_path.exists():
            cleaned.append(item)
        else:
            missing += 1

    out_path = base_dir / f'MSASL_{split}_clean.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f'\n{split.upper()}:')
    print(f'original: {len(data):,}')
    print(f'cleaned: {len(cleaned):,}')
    print(f'missing: {missing:,}')
    print(f'saved: {out_path}\n')


if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        clean_split(split)