#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import logging
import argparse
from datetime import datetime

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Мониторинг директории для автоматической транскрибации аудиофайлов")
    
    parser.add_argument("--watch-dir", default="/home/ubuntu/test_audio",
                        help="Директория для мониторинга")
    parser.add_argument("--processed-dir", default="/home/ubuntu/test_audio_complete",
                        help="Директория для обработанных файлов")
    parser.add_argument("--output-dir", default="/home/ubuntu/test_output",
                        help="Директория для результатов")
    parser.add_argument("--interval", type=int, default=30,
                        help="Интервал проверки в секундах")
    parser.add_argument("--model", default="small", 
                        choices=["tiny", "small", "medium", "large-v3"],
                        help="Размер модели Whisper")
    parser.add_argument("--left-lang", default="Русский",
                        choices=["Автоматическое определение", "Русский", "Казахский", "Английский"],
                        help="Язык для левого канала (менеджер)")
    parser.add_argument("--right-lang", default="Русский",
                        choices=["Автоматическое определение", "Русский", "Казахский", "Английский"],
                        help="Язык для правого канала (клиент)")
    
    return parser.parse_args()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_directories(dirs):
    """Создание директорий, если они не существуют"""
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Создана директория: {directory}")

def process_audio_file(file_path, output_dir, processed_dir, model, left_lang, right_lang):
    """Обработка аудиофайла через CLI-скрипт"""
    filename = os.path.basename(file_path)
    
    # Формируем команду с переданными параметрами - используем текущий интерпретатор Python
    cmd = [
        sys.executable,  # Используем текущий интерпретатор Python
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cli_transcribe.py"),
        "--input", os.path.dirname(file_path),
        "--output", output_dir,  # Используем одну и ту же выходную директорию
        "--model", model,
        "--left-lang", left_lang,
        "--right-lang", right_lang
    ]
    
    logging.info(f"Запуск транскрибации: {filename}")
    logging.info(f"Используемая модель: {model}")
    logging.info(f"Язык левого канала: {left_lang}")
    logging.info(f"Язык правого канала: {right_lang}")
    
    try:
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode == 0:
            # Перемещаем файл в директорию обработанных
            processed_path = os.path.join(processed_dir, filename)
            os.rename(file_path, processed_path)
            logging.info(f"Транскрибация завершена.")
            logging.info(f"Файл перемещен в {processed_path}")
            return True
        else:
            logging.error(f"Ошибка при транскрибации: {process.stderr}")
            return False
    except Exception as e:
        logging.error(f"Ошибка при запуске процесса: {e}")
        return False

def main():
    """Основная функция мониторинга"""
    args = parse_arguments()
    
    # Создаем необходимые директории
    setup_directories([args.watch_dir, args.processed_dir, args.output_dir])
    
    logging.info("Запущен мониторинг аудиофайлов...")
    logging.info(f"Директория мониторинга: {args.watch_dir}")
    logging.info(f"Директория обработанных файлов: {args.processed_dir}")
    logging.info(f"Директория результатов: {args.output_dir}")
    logging.info(f"Интервал проверки: {args.interval} секунд")
    logging.info(f"Модель: {args.model}")
    logging.info(f"Язык левого канала: {args.left_lang}")
    logging.info(f"Язык правого канала: {args.right_lang}")
    
    while True:
        for filename in os.listdir(args.watch_dir):
            file_path = os.path.join(args.watch_dir, filename)
            if os.path.isfile(file_path) and (
                filename.lower().endswith('.mp3') or
                filename.lower().endswith('.wav') or
                filename.lower().endswith('.flac') or
                filename.lower().endswith('.m4a') or
                filename.lower().endswith('.ogg')
            ):
                logging.info(f"Найден новый аудиофайл: {filename}")
                process_audio_file(
                    file_path, 
                    args.output_dir,
                    args.processed_dir,
                    args.model, 
                    args.left_lang, 
                    args.right_lang
                )
        
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
