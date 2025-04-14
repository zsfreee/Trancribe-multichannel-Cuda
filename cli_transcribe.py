#!/usr/bin/env python3
import os
import sys
import glob
import torch
import whisper
from pydub import AudioSegment
import pandas as pd
import argparse
import time

def transcribe_folder(input_dir, output_dir, params):
    print(f"CUDA доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Используемое устройство: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("GPU не доступен, используется CPU")
        device = torch.device("cpu")

    # Указываем директорию кэша явно
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "whisper"
    os.makedirs(str(cache_dir), exist_ok=True)

    # Загрузка модели
    print(f"Загрузка модели {params['model_size']}...")
    model = whisper.load_model(params['model_size'], download_root=str(cache_dir)).to(device)
    print("Модель успешно загружена на " + ("GPU" if torch.cuda.is_available() else "CPU"))

    # Поиск аудиофайлов
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))

    total_files = len(audio_files)
    if total_files == 0:
        print(f"Аудиофайлы не найдены в папке {input_dir}")
        return

    print(f"Найдено аудиофайлов: {total_files}")

    # Сопоставление языка с кодом для Whisper
    language_map = {
        "Автоматическое определение": None,
        "Русский": "ru",
        "Казахский": "kk",
        "Английский": "en"
    }

    # Создание структуры папок
    subdirs = {
        "manager": os.path.join(output_dir, "Только менеджер"),
        "client": os.path.join(output_dir, "Только клиент"),
        "with_time": os.path.join(output_dir, "С таймкодами"),
        "no_time": os.path.join(output_dir, "Без таймкодов")
    }

    for folder_path in subdirs.values():
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    csv_dir = os.path.join(output_dir, "CSV transcribe")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # Обработка файлов
    successful = 0
    failed = 0
    start_time = time.time()

    for i, audio_file in enumerate(audio_files):
        file_name = os.path.basename(audio_file).split('.')[0]
        print(f"\nОбработка файла {i+1} из {total_files}: {file_name}")
        file_start_time = time.time()

        try:
            # Разделение стерео-каналов
            audio = AudioSegment.from_file(audio_file)

            # Выделение каналов
            manager_channel = audio.split_to_mono()[0]  # Левый канал
            client_channel = audio.split_to_mono()[1]   # Правый канал

            # Сохранение каналов в отдельные файлы
            manager_audio_path = "manager_channel.wav"
            client_audio_path = "client_channel.wav"

            manager_channel.export(manager_audio_path, format="wav")
            client_channel.export(client_audio_path, format="wav")

            # Транскрибация канала менеджера
            print("Транскрибация канала менеджера...")

            transcribe_options_manager = {
                "temperature": params['left_temperature'],
                "beam_size": params['left_beam_size'],
                "patience": params['left_patience'],
                "best_of": int(params['left_best_of']),
                "no_speech_threshold": params['left_no_speech']
            }

            if language_map[params['left_language']]:
                transcribe_options_manager["language"] = language_map[params['left_language']]

            result_manager = model.transcribe(manager_audio_path, **transcribe_options_manager)

            # Транскрибация канала клиента
            print("Транскрибация канала клиента...")

            transcribe_options_client = {
                "temperature": params['right_temperature'],
                "beam_size": params['right_beam_size'],
                "patience": params['right_patience'],
                "best_of": int(params['right_best_of']),
                "no_speech_threshold": params['right_no_speech']
            }

            if language_map[params['right_language']]:
                transcribe_options_client["language"] = language_map[params['right_language']]

            result_client = model.transcribe(client_audio_path, **transcribe_options_client)

            # Сбор всех сегментов в один список
            all_segments = []

            # Добавление сегментов от менеджера
            for segment in result_manager["segments"]:
                all_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": "Менеджер",
                    "text": segment["text"]
                })

            # Добавление сегментов от клиента
            for segment in result_client["segments"]:
                all_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": "Клиент",
                    "text": segment["text"]
                })

            # Сортировка сегментов по времени
            all_segments.sort(key=lambda x: x["start"])

            # Сохранение результатов
            # Объединенный диалог с таймкодами
            combined_file_with_time = os.path.join(subdirs["with_time"], f"{file_name}.txt")
            with open(combined_file_with_time, "w", encoding="utf-8") as f:
                for segment in all_segments:
                    f.write(f"{segment['start']:.2f} - {segment['end']:.2f} | {segment['speaker']}: {segment['text']}\n")

            # Объединенный диалог без таймкодов
            combined_file = os.path.join(subdirs["no_time"], f"{file_name}.txt")
            with open(combined_file, "w", encoding="utf-8") as f:
                for segment in all_segments:
                    f.write(f"{segment['speaker']}: {segment['text']}\n")

            # Сохранение только реплик менеджера
            manager_file = os.path.join(subdirs["manager"], f"{file_name}.txt")
            with open(manager_file, "w", encoding="utf-8") as f:
                for segment in result_manager["segments"]:
                    f.write(f"{segment['text']}\n")

            # Сохранение только реплик клиента
            client_file = os.path.join(subdirs["client"], f"{file_name}.txt")
            with open(client_file, "w", encoding="utf-8") as f:
                for segment in result_client["segments"]:
                    f.write(f"{segment['text']}\n")

            # Удаление временных файлов
            os.remove(manager_audio_path)
            os.remove(client_audio_path)

            successful += 1
            file_duration = time.time() - file_start_time
            print(f"✅ Файл обработан успешно: {file_name} (за {file_duration:.2f} сек)")

        except Exception as e:
            failed += 1
            print(f"❌ Ошибка при обработке файла {file_name}: {str(e)}")

        # Отображение прогресса
        progress = int(((i + 1) / total_files) * 100)
        print(f"Прогресс: {progress}%")

    # Создание CSV файлов
    print("\nСоздание CSV файлов...")

    # Функция для чтения содержимого текстового файла
    def read_file_content(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {str(e)}")
            return ""

    # Создание отдельных CSV для каждой папки
    for folder_name, folder_path in subdirs.items():
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

        if not txt_files:
            print(f"В папке {folder_path} нет текстовых файлов.")
            continue

        folder_label = os.path.basename(folder_path)
        csv_filename = os.path.join(folder_path, f"{folder_label}.csv")

        data = []
        for txt_file in txt_files:
            file_name = os.path.basename(txt_file).split('.')[0]
            content = read_file_content(txt_file)
            data.append({"Файл": file_name, "Транскрибация": content})

        df = pd.DataFrame(data)
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"Создан CSV файл для папки {folder_label}")

    # Создание общего CSV файла
    all_files = set()
    for folder_path in subdirs.values():
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        for txt_file in txt_files:
            all_files.add(os.path.basename(txt_file).split('.')[0])

    combined_data = []
    for file_name in all_files:
        row = {"Файл": file_name}

        for folder_path in subdirs.values():
            folder_label = os.path.basename(folder_path)
            txt_path = os.path.join(folder_path, f"{file_name}.txt")

            if os.path.exists(txt_path):
                content = read_file_content(txt_path)
                row[folder_label] = content
            else:
                row[folder_label] = ""

        combined_data.append(row)

    combined_csv_path = os.path.join(csv_dir, "all_transcriptions.csv")
    df_combined = pd.DataFrame(combined_data)
    df_combined.to_csv(combined_csv_path, index=False, encoding='utf-8-sig')

    print(f"Создан общий CSV файл: {combined_csv_path}")

    # Итоги
    total_time = time.time() - start_time
    print("\n===== Итоги обработки =====")
    print(f"Всего файлов: {total_files}")
    print(f"Успешно обработано: {successful}")
    print(f"С ошибками: {failed}")
    print(f"Общее время выполнения: {total_time:.2f} сек ({total_time/60:.2f} мин)")
    print("Транскрибация завершена")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Whisper Transcription CLI Tool')
    parser.add_argument('--input', '-i', required=True, help='Input directory with audio files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    parser.add_argument('--model', '-m', default='small', choices=['tiny', 'small', 'medium', 'large-v3'],
                        help='Whisper model size')
    parser.add_argument('--left-lang', default='Русский',
                        choices=['Автоматическое определение', 'Русский', 'Казахский', 'Английский'],
                        help='Language for left channel (manager)')
    parser.add_argument('--right-lang', default='Русский',
                        choices=['Автоматическое определение', 'Русский', 'Казахский', 'Английский'],
                        help='Language for right channel (client)')

    args = parser.parse_args()

    # Параметры по умолчанию
    params = {
        'model_size': args.model,
        'left_language': args.left_lang,
        'left_temperature': 0.2,
        'left_beam_size': 5,
        'left_patience': 1.0,
        'left_best_of': '5',
        'left_no_speech': 0.6,
        'right_language': args.right_lang,
        'right_temperature': 0.2,
        'right_beam_size': 5,
        'right_patience': 1.0,
        'right_best_of': '5',
        'right_no_speech': 0.6
    }

    transcribe_folder(args.input, args.output, params)
