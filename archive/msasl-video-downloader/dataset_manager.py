import os
import shutil
import json
import time
import yt_dlp
from moviepy.editor import *


class DatasetManager:
    _singleton = None

    def __new__(cls, *args, **kwargs):
        if not cls._singleton:
            cls._singleton = super(DatasetManager, cls).__new__(cls, *args, **kwargs)
        return cls._singleton

    def __init__(self):
        self.dataset_configs = {
            'train': {'base_path': '../ms_asl/videos/train', 'json_path':'../ms_asl/MSASL_train.json'},
            'val': {
                'base_path': '../ms_asl/videos/val', 'json_path':'../ms_asl/MSASL_val.json'},
            'test': {'base_path': '../ms_asl/videos/test', 'json_path': '../ms_asl/MSASL_test.json'}
        }

    def isValidDataset(self, dataset):
        return True if dataset in [1, 2, 3, 4, 5] else False

    def datasetSize(self, dataset):
        if dataset == 1:
            return 100
        elif dataset == 2:
            return 200
        elif dataset == 3:
            return 500
        elif dataset == 4:
            return 1000
        else:  # dataset = 5 yani bütün ms-asl1000 verisetini indirmek için ekledim..!
            return 'all'

    def createDirectory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f'{directory} dizini oluşturuldu.')
            return True
        else:
            print(f'{directory} zaten mevcut.')
            return False

    def deleteDirectory(self, directory):
        if os.path.exists(directory):
            if len(os.listdir(directory)) == 0:
                os.rmdir(directory)
            else:
                shutil.rmtree(directory)
            print(f'{directory} silindi.')
        else:
            print(f'{directory} mevcut değil.')

    def downloadAndTrimSplit(self, split_name, dataset_size, size_label):
        """Tek bir split (train/val/test) için videoları indirir..!"""
        config = self.dataset_configs[split_name]
        json_path = config['json_path']
        classes_path = '../ms_asl/MSASL_classes.json'

        base_path = config['base_path']
        downloadDirectory = os.path.join(base_path, size_label)

        print(f"\n{'=' * 60}")
        print(f"Split: {split_name.upper()}")
        print(f"Boyut: {dataset_size if dataset_size != 'all' else 'TÜM VIDEOLAR'}")
        print(f"Dizin: {downloadDirectory}")
        print(f"JSON: {json_path}")
        print(f"{'=' * 60}\n")

        if not os.path.exists(json_path):
            print(f'HATA: {json_path} bulunamadı!')
            return 0, 0

        if not os.path.exists(classes_path):
            print(f'HATA: {classes_path} bulunamadı!')
            return 0, 0

        with open(json_path, 'r') as f:
            videos = json.load(f)

        with open(classes_path, 'r') as f:
            words = json.load(f)

        os.makedirs(downloadDirectory, exist_ok=True)

        total_available = len(videos)
        print(f'JSONda toplam {total_available} video mevcut')

        if dataset_size == 'all':
            actual_size = total_available
            print(f'TÜM VIDEOLAR İNDİRİLECEK: {actual_size} video\n')
        elif dataset_size > total_available:
            print(f'UYARI: İstenen {dataset_size} video, mevcut {total_available} videodan fazla!')
            actual_size = total_available
            print(f'{actual_size} video indirilecek.\n')
        else:
            actual_size = dataset_size
            print(f'{actual_size} video indirilecek.\n')

        success_count = 0
        fail_count = 0

        for i in range(actual_size):
            url = videos[i]['url']
            start_time = videos[i]['start_time']
            end_time = videos[i]['end_time']
            label = videos[i]['label']

            title = words[label]
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            video_name = f'{safe_title}_{i}.mp4'
            video_path = os.path.join(downloadDirectory, video_name)
            temp_path = f'temp_{i}.mp4'

            # Retry mekanizması: 3 deneme
            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    if retry_count > 0:
                        print(f'Tekrar deneniyor ({retry_count}/{max_retries})...')
                        time.sleep(5)

                    print(f'[{i + 1}/{actual_size}] İndiriliyor: {video_name}')

                    ydl_opts = {
                        'format': 'best[ext=mp4]',
                        'outtmpl': temp_path,
                        'quiet': True,
                        'no_warnings': True,
                        'socket_timeout': 60,  # timeout'u 60 yaptım
                        'retries': 10,  # yt-dlp kendi retry'ları

                        'nopart': True, # yarım kalan .part dosyası oluşturmasın..! (yoksa patlıyor)
                        'continuedl': False, # kaldığı yerden devam etmesin
                        'overwrites': True, # üstüne yaz
                        'fragment_retries': 10,
                        'concurrent_fragment_downloads': 1
                    }

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])

                    clip = VideoFileClip(temp_path).subclip(start_time, end_time)
                    clip.write_videofile(video_path, codec='libx264', audio=False, verbose=False, logger=None)

                    clip.close()

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    print(f'Başarılı!\n')
                    success_count += 1
                    success = True

                except Exception as e:
                    error_message = str(e).lower()

                    if 'private' in error_message or 'unavailable' in error_message:
                        print(f'Atlandı (private/unavailable): {url}\n')
                        fail_count += 1
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        break

                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f'{max_retries} denemeden sonra başarısız: {e}\n')
                        fail_count += 1
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    else:
                        print(f'Hata: {e}')

                if os.path.exists(temp_path) and not success:
                    os.remove(temp_path)

            # youtubeun ratelimitini önlemek için her 50 videoda 30 saniye bekliyor..!
            if (i + 1) % 50 == 0:
                print(f"\n50 video indirildi. Rate limiting'i önlemek için 30 saniye bekleniyor...")
                print(f'İlerleme: {i + 1}/{actual_size} ({(i + 1) / actual_size * 100:.1f}%)\n')
                time.sleep(30)

        print(f"\n{'=' * 60}")
        print(f"{split_name.upper()} tamamlandı!")
        print(f"Dizin: {downloadDirectory}")
        print(f"✓ Başarılı: {success_count} | ✗ Başarısız: {fail_count}")
        print(f"{'=' * 60}\n")

        return success_count, fail_count

    def downloadAllSplits(self, dataset_size):
        """Train, Val, Test için tüm videoları indirme..!"""

        # Boyut etiketini belirle
        if dataset_size == 'all':
            size_label = 'MS-ASL-ALL'
        else:
            size_label = f'MS-ASL{dataset_size}'

        print(f"\n{'#' * 60}")
        print(f'TÜM SPLİTLER İÇİN İNDİRME BAŞLIYOR')
        print(f'Dataset Boyutu: {size_label}')
        print(f"{'#' * 60}\n")

        if dataset_size == 'all':
            print('UYARI: Tüm videoları indiriyorsunuz!')
            print('Bu işlem çok uzun sürebilir..!')
            confirm = input("Devam etmek istiyor musunuz? (yes/no): ")
            if confirm.lower() not in ['yes', 'y']:
                print("İşlem iptal edildi.")
                return

        total_success = 0
        total_fail = 0

        # TRAIN videolarını indir
        print('\nTRAIN videoları indiriliyor...')
        success, fail = self.downloadAndTrimSplit('train', dataset_size, size_label)
        total_success += success
        total_fail += fail

        # VAL videolarını indir
        print('\nVAL videoları indiriliyor...')
        success, fail = self.downloadAndTrimSplit('val', dataset_size, size_label)
        total_success += success
        total_fail += fail

        # TEST videolarını indir
        print('\nTEST videoları indiriliyor...')
        success, fail = self.downloadAndTrimSplit('test', dataset_size, size_label)
        total_success += success
        total_fail += fail

        # özet kısmı
        print(f"\n{'#' * 60}")
        print(f"TÜM İNDİRMELER TAMAMLANDI!")
        print(f"{'#' * 60}")
        print(f"Toplam Başarılı: {total_success}")
        print(f"Toplam Başarısız: {total_fail}")
        print(f"{'#' * 60}\n")

    def generateDataset(self):
        print()
        print(15 * '-', 'DATASET BOYUTU SEÇİN', 15 * '-')
        print('[1] MS-ASL100  (Her split için 100 video)')
        print('[2] MS-ASL200  (Her split için 200 video)')
        print('[3] MS-ASL500  (Her split için 500 video)')
        print('[4] MS-ASL1000 (Her split için 1000 video)')
        print('[5] MS-ASL-ALL (Her split için TÜM videolar)')
        print(53 * '-')
        print('\nNot: Bu seçim train, val ve test için AYNI ANDA uygulanır!')

        answer = int(input('Seçiminizi girin [1-5]: '))

        if not self.isValidDataset(answer):
            print('Geçersiz seçim. Lütfen tekrar deneyin.')
            return

        dataset_size = self.datasetSize(answer)

        # Dizinleri oluştur
        if dataset_size == 'all':
            size_label = 'MS-ASL-ALL'
        else:
            size_label = f'MS-ASL{dataset_size}'

        # Tüm dizinleri kontrol et
        all_exist = True
        for split_name, config in self.dataset_configs.items():
            target_dir = os.path.join(config['base_path'], size_label)
            if not os.path.exists(target_dir):
                all_exist = False
                break

        if all_exist:
            print(f'\n{size_label} dizinleri zaten mevcut!')
            print('Önce dizinleri silmek ister misiniz? (Dataset Manager menüsünden "Delete a dataset" seçin!)\n')
            return

        # Dizinleri oluştur
        for split_name, config in self.dataset_configs.items():
            target_dir = os.path.join(config['base_path'], size_label)
            self.createDirectory(target_dir)

        # İndirmeyi başlat
        self.downloadAllSplits(dataset_size)

    def deleteDataset(self):
        print()
        print(15 * '-', 'SİLİNECEK DATASET BOYUTU', 15 * '-')
        print('[1] MS-ASL100')
        print('[2] MS-ASL200')
        print('[3] MS-ASL500')
        print('[4] MS-ASL1000')
        print('[5] MS-ASL-ALL')
        print(50 * '-')
        print('\nNot: Bu seçim train, val ve test dizinlerini SİLER!\n')

        answer = int(input('Seçiminizi girin [1-5]: '))

        if not self.isValidDataset(answer):
            print('Geçersiz seçim.')
            return

        dataset_size = self.datasetSize(answer)

        if dataset_size == 'all':
            size_label = 'MS-ASL-ALL'
        else:
            size_label = f'MS-ASL{dataset_size}'

        print(f'\nUYARI: Aşağıdaki dizinler silinecek:')
        for split_name, config in self.dataset_configs.items():
            target_dir = os.path.join(config['base_path'], size_label)
            print(f'   - {target_dir}')

        confirm = input('\nDevam etmek istiyor musunuz? (evet/hayir): ')
        if confirm.lower() not in ['evet', 'e', 'yes', 'y']:
            print('İşlem iptal edildi.')
            return

        for split_name, config in self.dataset_configs.items():
            target_dir = os.path.join(config['base_path'], size_label)
            self.deleteDirectory(target_dir)

        print(f'\n{size_label} dizinleri silindi.\n')

    def menu(self):
        print()
        print(15 * '-', 'DATASET MANAGER', 15 * '-')
        print('[1] Generate dataset (train + val + test)')
        print('[2] Delete dataset')
        print('[3] Back to main menu')
        print(50 * '-')