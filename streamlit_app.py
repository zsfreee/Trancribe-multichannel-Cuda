import streamlit as st
import os
import whisper
import torch
from pydub import AudioSegment
import glob
import pandas as pd
import time
import tempfile
import zipfile
import io
import threading
import base64
import shutil
import subprocess

# Функция для проверки наличия FFmpeg
def is_tool_installed(name):
    """Проверяет, установлен ли инструмент в системе."""
    return shutil.which(name) is not None

# Функция для отображения инструкций по установке FFmpeg
def show_ffmpeg_installation_instructions():
    st.error("⚠️ FFmpeg не обнаружен в вашей системе!")
    st.markdown("""
    ### Для корректной работы с аудиофайлами необходимо установить FFmpeg:
    
    #### Windows:
    1. Скачайте FFmpeg с официального сайта: [FFmpeg Downloads](https://ffmpeg.org/download.html)
    2. Распакуйте архив и добавьте путь к папке bin в переменную среды PATH
    
    #### Mac OS:
    ```
    brew install ffmpeg
    ```
    
    #### Linux (Ubuntu/Debian):
    ```
    sudo apt update
    sudo apt install ffmpeg
    ```
    
    После установки перезапустите приложение.
    """)

# Функция для создания ссылки на скачивание
def get_binary_file_downloader_html(bin_file, file_label='Файл'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Скачать {file_label}</a>'
    return href

# Функция для создания zip-архива
def create_zip_file(directory):
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory)
                zipf.write(file_path, arcname)
    memory_file.seek(0)
    return memory_file

# Транскрибация аудио
def transcribe_audio(input_files, params, progress_bar, status_text, log_area):
    try:
        # Проверка доступности GPU
        log_area.markdown(f"CUDA доступен: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log_area.markdown(f"Используемое устройство: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
        else:
            log_area.markdown("GPU не доступен, используется CPU")
            device = torch.device("cpu")
        
        # Указываем директорию кэша явно
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "whisper"
        os.makedirs(str(cache_dir), exist_ok=True)
        
        # Загрузка модели
        log_area.markdown(f"Загрузка модели {params['model_size']}...")
        model = whisper.load_model(params['model_size'], download_root=str(cache_dir)).to(device)
        log_area.markdown("Модель успешно загружена на " + ("GPU" if torch.cuda.is_available() else "CPU"))
        
        # Создание временной папки для результатов
        temp_output_dir = tempfile.mkdtemp()
        
        # Создание структуры папок
        subdirs = {
            "manager": os.path.join(temp_output_dir, "Только менеджер"),
            "client": os.path.join(temp_output_dir, "Только клиент"),
            "with_time": os.path.join(temp_output_dir, "С таймкодами"),
            "no_time": os.path.join(temp_output_dir, "Без таймкодов")
        }
        
        for folder_path in subdirs.values():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        csv_dir = os.path.join(temp_output_dir, "CSV transcribe")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        # Сопоставление языка с кодом для Whisper
        language_map = {
            "Автоматическое определение": None,
            "Русский": "ru",
            "Казахский": "kk",
            "Английский": "en"
        }
        
        # Обработка файлов
        total_files = len(input_files)
        successful = 0
        failed = 0
        
        for i, uploaded_file in enumerate(input_files):
            # Сохраняем загруженный файл во временную директорию
            file_name = uploaded_file.name.split('.')[0]
            temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            log_area.markdown(f"\nОбработка файла {i+1} из {total_files}: {file_name}")
            status_text.text(f"Обработка: {i+1}/{total_files}")
            
            try:
                # Разделение стерео-каналов
                audio = AudioSegment.from_file(temp_file_path)
                
                # Выделение каналов
                manager_channel = audio.split_to_mono()[0]  # Левый канал
                client_channel = audio.split_to_mono()[1]   # Правый канал
                
                # Сохранение каналов в отдельные файлы
                manager_audio_path = os.path.join(tempfile.gettempdir(), "manager_channel.wav")
                client_audio_path = os.path.join(tempfile.gettempdir(), "client_channel.wav")
                
                manager_channel.export(manager_audio_path, format="wav")
                client_channel.export(client_audio_path, format="wav")
                
                # Транскрибация канала менеджера
                log_area.markdown("Транскрибация канала менеджера...")
                
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
                log_area.markdown("Транскрибация канала клиента...")
                
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
                log_area.markdown(f"✅ Файл обработан успешно: {file_name}")
                
            except Exception as e:
                failed += 1
                log_area.markdown(f"❌ Ошибка при обработке файла {file_name}: {str(e)}")
            
            # Обновление прогресса
            progress_value = int(((i + 1) / total_files) * 100)
            progress_bar.progress(progress_value / 100)
        
        # Создание CSV файлов
        log_area.markdown("\nСоздание CSV файлов...")
        
        # Функция для чтения содержимого текстового файла
        def read_file_content(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                log_area.markdown(f"Ошибка при чтении файла {file_path}: {str(e)}")
                return ""
        
        # Создание отдельных CSV для каждой папки
        for folder_name, folder_path in subdirs.items():
            txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
            
            if not txt_files:
                log_area.markdown(f"В папке {folder_path} нет текстовых файлов.")
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
            log_area.markdown(f"Создан CSV файл для папки {folder_label}")
        
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
        
        log_area.markdown(f"Создан общий CSV файл: {combined_csv_path}")
        
        # Итоги
        log_area.markdown("\n===== Итоги обработки =====")
        log_area.markdown(f"Всего файлов: {total_files}")
        log_area.markdown(f"Успешно обработано: {successful}")
        log_area.markdown(f"С ошибками: {failed}")
        status_text.text("Транскрибация завершена")
        
        # Создание zip-архива с результатами
        zip_buffer = create_zip_file(temp_output_dir)
        
        return {
            "status": "success",
            "output_dir": temp_output_dir,
            "zip_buffer": zip_buffer
        }
        
    except Exception as e:
        log_area.markdown(f"Ошибка в процессе транскрибации: {str(e)}")
        status_text.text("Ошибка")
        return {
            "status": "error",
            "message": str(e)
        }

# Функция для проверки CUDA
def check_cuda(log_area):
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            log_area.markdown(f"CUDA доступен. Обнаружено устройство: {device_name}")
            log_area.markdown(f"Версия CUDA: {cuda_version}")
        else:
            log_area.markdown("CUDA недоступен. Будет использоваться CPU.")
            log_area.markdown("Для ускорения работы рекомендуется установить PyTorch с поддержкой CUDA.")
    except Exception as e:
        log_area.markdown(f"Ошибка при проверке CUDA: {str(e)}")

def main():
    st.set_page_config(
        page_title="Whisper Transcription Tool",
        page_icon="🎙️",
        layout="wide"
    )
    
    # Определяем, доступна ли CUDA
    cuda_available = torch.cuda.is_available()
    device_info = f"🖥️ CPU" if not cuda_available else f"🚀 CUDA GPU: {torch.cuda.get_device_name(0)}"
    
    # Заголовок приложения с информацией об используемом устройстве
    st.title(f"Whisper Transcription Tool [{device_info}]")
    st.markdown("### Инструмент для транскрибации многоканального аудио")
    
    if 'transcription_result' not in st.session_state:
        st.session_state.transcription_result = None
    
    # Стили для вкладок с адаптивным дизайном
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    /* Стили для десктопов (по умолчанию) */
    .stTabs [data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: 600;
        height: 50px;
        padding: 0 15px;
    }
    .stTabs [role="tab"] p {
        font-size: 20px !important;
    }
    
    /* Стили для планшетов */
    @media (max-width: 992px) {
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 18px !important;
            height: 45px;
            padding: 0 10px;
        }
        .stTabs [role="tab"] p {
            font-size: 18px !important;
        }
    }
    
    /* Стили для мобильных устройств */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 14px !important;
            height: 40px;
            padding: 0 5px;
        }
        .stTabs [role="tab"] p {
            font-size: 14px !important;
        }
        
        /* Адаптивные стили для таблицы в справке */
        table {
            font-size: 12px !important;
            width: 100% !important;
            table-layout: fixed !important;
        }
        th, td {
            padding: 5px 2px !important;
            word-wrap: break-word !important;
            white-space: normal !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Создаем вкладки с тематическими смайликами (обычный текст)
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Файлы", "⚙️ Параметры", "▶️ Выполнение", "ℹ️ Справка"])
    
    with tab1:
        st.header("📁 Загрузка аудиофайлов")
        uploaded_files = st.file_uploader("Выберите аудиофайлы (MP3, WAV, M4A, FLAC, OGG)", 
                                          type=["mp3", "wav", "m4a", "flac", "ogg"],
                                          accept_multiple_files=True)
    
    with tab2:
        st.header("⚙️ Параметры транскрибации")
        
        # Выбор модели
        st.subheader("🤖 Модель Whisper")
        model_size = st.selectbox("Размер модели:", 
                                  ["tiny", "small", "medium", "large-v3"],
                                  index=3)  # По умолчанию large-v3
        
        # Пустая строка для создания отступа
        st.markdown("")
        
        # Параметры для левого и правого каналов в двух колонках
        st.subheader("Настройка параметров каналов")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👨‍💼 Параметры для левого канала (Менеджер)")
            
            left_language = st.selectbox(
                "Язык (левый канал):",
                ["Автоматическое определение", "Русский", "Казахский", "Английский"],
                index=1,
                key="left_lang"
            )
            
            left_temperature = st.slider("Температура (левый канал):", 0.0, 1.0, 0.2, 0.1)
            left_beam_size = st.slider("Beam size (левый канал):", 1, 10, 5, 1)
            left_patience = st.slider("Patience (левый канал):", 0.5, 2.0, 1.0, 0.1)
            left_best_of = st.selectbox(
                "Best of (левый канал):",
                ["1", "3", "5", "7", "10"],
                index=2,
                key="left_best_of"
            )
            left_no_speech = st.slider("No speech threshold (левый канал):", 0.1, 1.0, 0.6, 0.1)
        
        with col2:
            st.markdown("### 👨‍👩‍👧 Параметры для правого канала (Клиент)")
            
            right_language = st.selectbox(
                "Язык (правый канал):",
                ["Автоматическое определение", "Русский", "Казахский", "Английский"],
                index=1,
                key="right_lang"
            )
            
            right_temperature = st.slider("Температура (правый канал):", 0.0, 1.0, 0.2, 0.1)
            right_beam_size = st.slider("Beam size (правый канал):", 1, 10, 5, 1)
            right_patience = st.slider("Patience (правый канал):", 0.5, 2.0, 1.0, 0.1)
            right_best_of = st.selectbox(
                "Best of (правый канал):",
                ["1", "3", "5", "7", "10"],
                index=2,
                key="right_best_of"
            )
            right_no_speech = st.slider("No speech threshold (правый канал):", 0.1, 1.0, 0.6, 0.1)
    
    with tab3:
        st.header("▶️ Выполнение транскрибации")
        
        # Создаем области для логов и статуса
        log_area = st.empty()
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Проверяем CUDA при первой загрузке страницы
        if 'cuda_checked' not in st.session_state:
            check_cuda(log_area)
            st.session_state.cuda_checked = True
        
        # Проверяем наличие FFmpeg
        if not is_tool_installed("ffmpeg"):
            show_ffmpeg_installation_instructions()
        else:
            # Кнопка запуска транскрибации
            col1, col2 = st.columns([1, 3])
            with col1:
                start_button = st.button("🚀 Начать транскрибацию", type="primary")
            
            # Если кнопка нажата и файлы загружены
            if start_button:
                if not uploaded_files:
                    st.error("❌ Выберите аудиофайлы для транскрибации!")
                else:
                    # Собираем параметры
                    params = {
                        'model_size': model_size,
                        'left_language': left_language,
                        'left_temperature': left_temperature,
                        'left_beam_size': left_beam_size,
                        'left_patience': left_patience,
                        'left_best_of': left_best_of,
                        'left_no_speech': left_no_speech,
                        'right_language': right_language,
                        'right_temperature': right_temperature,
                        'right_beam_size': right_beam_size,
                        'right_patience': right_patience,
                        'right_best_of': right_best_of,
                        'right_no_speech': right_no_speech
                    }
                    
                    # Очищаем log_area перед запуском
                    log_area = st.empty()
                    with st.spinner("⏳ Выполняется транскрибация..."):
                        status_text.text("🔄 Подготовка к транскрибации...")
                        st.session_state.transcription_result = transcribe_audio(
                            uploaded_files, params, progress_bar, status_text, log_area
                        )
            
            # Если транскрибация завершена, показываем результаты
            if st.session_state.transcription_result and st.session_state.transcription_result['status'] == 'success':
                st.success("✅ Транскрибация успешно завершена!")
                
                # Скачивание результатов
                st.subheader("📥 Скачать результаты")
                
                # Конвертируем буфер в байты для скачивания
                if 'zip_buffer' in st.session_state.transcription_result:
                    zip_data = st.session_state.transcription_result['zip_buffer'].getvalue()
                    st.download_button(
                        label="📦 Скачать все результаты (ZIP)",
                        data=zip_data,
                        file_name="transcription_results.zip",
                        mime="application/zip"
                    )
    
    with tab4:
        st.header("ℹ️ Справка")
        
        st.markdown("""
        ## Описание параметров Whisper
        
        ### Выбор модели Whisper
        
        Whisper предлагает несколько размеров модели, отличающихся по точности и скорости:
        
        | Модель | Размер | Требования к VRAM | Качество | Скорость |
        |--------|--------|-------------------|----------|----------|
        | tiny | ~40MB | ~1GB | Базовое | Очень быстро |
        | small | ~500MB | ~2GB | Очень хорошее | Средне |
        | medium | ~1.5GB | ~5GB | Отличное | Медленно |
        | large-v3 | ~3GB | ~10GB | Наилучшее | Очень медленно |
        
        ### Параметры для каналов
        
        Для каждого канала (левый - менеджер, правый - клиент) можно настроить следующие параметры:
        
        #### Язык
        Выбор языка для распознавания речи:
        
        - Автоматическое определение: Whisper самостоятельно определит язык
        - Русский: Оптимизировано для русской речи
        - Казахский: Оптимизировано для казахской речи
        - Английский: Оптимизировано для английской речи
        
        #### Температура (0-1)
        Влияет на случайность при декодировании текста:
        
        - 0.0: Наиболее детерминированный результат (точнее для четкой речи)
        - 0.2-0.4: Рекомендуемые значения для большинства случаев
        - более 0.5: Более разнообразные результаты (может помочь при нечеткой речи)
        
        #### Beam size (1-10)
        Количество лучей при поиске лучшей транскрипции:
        
        - 1: Самый быстрый, но менее точный (жадный поиск)
        - 5: Рекомендуемое значение (баланс между скоростью и точностью)
        - более 5: Более точно, но значительно медленнее
        
        #### Patience (0.5-2.0)
        Коэффициент терпения при поиске:
        
        - 1.0: Стандартное значение
        - менее 1.0: Быстрее, но менее точно
        - более 1.0: Медленнее, но потенциально более точно
        
        #### Best of (1, 3, 5, 7, 10)
        Количество сэмплов для выбора лучшего результата:
        
        - 1: Самый быстрый, но менее точный
        - 5: Рекомендуемое значение (хороший баланс)
        - более 5: Более точно, но требует больше ресурсов
        
        #### No speech threshold (0.1-1.0)
        Порог для определения отсутствия речи:
        
        - 0.1-0.3: Меньше пропусков тихой речи, но больше ложных срабатываний
        - 0.6: Рекомендуемое значение
        - более 0.7: Меньше ложных срабатываний, но больше риск пропустить тихую речь
        
        ## Рекомендуемые настройки для разных сценариев
        
        ### Для качественных записей с четкой речью (колл-центр)
        - Модель: large-v3 или medium
        - Язык: Соответствующий язык разговора
        - Температура: 0.0-0.2
        - Beam size: 5
        - Patience: 1.0
        - Best of: 5
        - No speech threshold: 0.6
        
        ### Для записей с шумом или нечеткой речью
        - Модель: large-v3
        - Язык: Соответствующий язык разговора
        - Температура: 0.3-0.5
        - Beam size: 5-8
        - Patience: 1.2-1.5
        - Best of: 7-10
        - No speech threshold: 0.4-0.5
        
        ### Для быстрой предварительной транскрибации
        - Модель: small или tiny
        - Язык: Соответствующий язык разговора
        - Температура: 0.0
        - Beam size: 1-3
        - Patience: 1.0
        - Best of: 1-3
        - No speech threshold: 0.6
        """)
        
if __name__ == "__main__":
    main()
````
