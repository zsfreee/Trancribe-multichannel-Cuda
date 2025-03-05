# Whisper Transcription Tool

Инструмент для транскрибации стерео-аудиофайлов с разделением на каналы для отдельной обработки речи менеджера и клиента. Приложение автоматически разделяет стерео-записи телефонных разговоров на отдельные каналы и создает текстовую транскрипцию с учетом говорящих.

## Особенности

- Автоматическое разделение стерео-записи на два канала (левый - менеджер, правый - клиент)
- Отдельная транскрибация для каждого канала с индивидуальными настройками
- Создание объединенного диалога с указанием времени
- Сохранение результатов в текстовом и CSV форматах
- Удобный графический интерфейс для настройки параметров
- Поддержка русского, казахского и английского языков
- Автоматическое определение и использование GPU (CUDA) для ускорения транскрибации
- Возможность остановки процесса транскрибации

## Требования

- Python 3.12.7
- FFmpeg (требуется для обработки аудио)
- NVIDIA GPU с CUDA для ускорения (опционально)
- Git https://git-scm.com/downloads/

## Установка

### 1. Клонируйте репозиторий:
```
git clone https://github.com/ваш-username/whisper-transcription-tool.git
cd whisper-transcription-tool
```

### 2. Создайте виртуальное окружение и активируйте его:
```
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Установите PyTorch с поддержкой CUDA (для использования GPU):
Не забываем обновить pip
```
python.exe -m pip install --upgrade pip
```
Установка PyTorch
```
# Для NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Без GPU:
pip install torch torchvision torchaudio
```

### 4. Установите остальные зависимости:
```
pip install -r requirements.txt
```

Если ошибка в requirements с torchvision
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/whisper.git
pip install pydub pandas soundfile PyQt6
```
### 5. Установите FFmpeg:

**Windows:**
- Скачайте FFmpeg с [официального сайта](https://ffmpeg.org/download.html) или с [GitHub](https://github.com/BtbN/FFmpeg-Builds/releases) (выберите версию с "shared" в названии)
- Распакуйте архив
- Скопируйте все файлы .exe и .dll из папки bin в корень проекта

Например, для Windows 64bit - необходимо перенести в корневую папку проекта следующие файлы из папки bin ffmpeg:
- avcodec-61.dll
- avdevice-61.dll
- avfilter-10.dll
- avformat-61.dll
- avutil-59.dll
- postproc-58.dll
- swresample-5.dll
- swscale-8.dll
- ffmpeg.exe
- ffplay.exe
- ffprobe.exe

**Linux:**
```
sudo apt-get update
sudo apt-get install ffmpeg
```

**Mac:**
```
brew install ffmpeg
```

## Использование

1. Запустите приложение:
```
python whisper_app.py
```

2. В интерфейсе приложения:
   - На вкладке "Файлы" выберите папку с аудиофайлами для обработки и папку для сохранения результатов
   - На вкладке "Параметры" настройте параметры модели Whisper для каждого канала
   - На вкладке "Выполнение" нажмите "Начать транскрибацию" и следите за процессом в журнале

3. После завершения транскрибации в выбранной папке будут созданы следующие подпапки:
   - "Только менеджер" - содержит транскрибацию левого канала
   - "Только клиент" - содержит транскрибацию правого канала
   - "С таймкодами" - содержит объединенный диалог с временными метками
   - "Без таймкодов" - содержит объединенный диалог без временных меток (оптимально для LLM)
   - "CSV transcribe" - содержит CSV-файлы со всеми типами транскрибаций

## Описание параметров

### Выбор модели Whisper

Whisper предлагает несколько размеров модели, отличающихся по точности и скорости:

| Модель    | Размер   | Требования к VRAM | Качество      | Скорость    |
|-----------|----------|-------------------|---------------|-------------|
| tiny      | ~40MB    | ~1GB              | Базовое       | Очень быстро|
| base      | ~150MB   | ~1GB              | Хорошее       | Быстро      |
| small     | ~500MB   | ~2GB              | Очень хорошее | Средне      |
| medium    | ~1.5GB   | ~5GB              | Отличное      | Медленно    |
| large-v3  | ~3GB     | ~10GB             | Наилучшее     | Очень медленно |

### Параметры для каналов

Для каждого канала (левый - менеджер, правый - клиент) можно настроить следующие параметры:

#### Язык
Выбор языка для распознавания речи:
- **Автоматическое определение**: Whisper самостоятельно определит язык
- **Русский**: Оптимизировано для русской речи
- **Казахский**: Оптимизировано для казахской речи
- **Английский**: Оптимизировано для английской речи

#### Температура (0-1)
Влияет на случайность при декодировании текста:
- **0.0**: Наиболее детерминированный результат (точнее для четкой речи)
- **0.2-0.4**: Рекомендуемые значения для большинства случаев
- **>0.5**: Более разнообразные результаты (может помочь при нечеткой речи)

#### Beam size (1-10)
Количество лучей при поиске лучшей транскрипции:
- **1**: Самый быстрый, но менее точный (жадный поиск)
- **5**: Рекомендуемое значение (баланс между скоростью и точностью)
- **>5**: Более точно, но значительно медленнее

#### Patience (0.5-2.0)
Коэффициент терпения при поиске:
- **1.0**: Стандартное значение
- **<1.0**: Быстрее, но менее точно
- **>1.0**: Медленнее, но потенциально более точно

#### Best of (1, 3, 5, 7, 10)
Количество сэмплов для выбора лучшего результата:
- **1**: Самый быстрый, но менее точный
- **5**: Рекомендуемое значение (хороший баланс)
- **>5**: Более точно, но требует больше ресурсов

#### No speech threshold (0.1-1.0)
Порог для определения отсутствия речи:
- **0.1-0.3**: Меньше пропусков тихой речи, но больше ложных срабатываний
- **0.6**: Рекомендуемое значение
- **>0.7**: Меньше ложных срабатываний, но больше риск пропустить тихую речь

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
- Модель: small или base
- Язык: Соответствующий язык разговора
- Температура: 0.0
- Beam size: 1-3
- Patience: 1.0
- Best of: 1-3
- No speech threshold: 0.6

## Проблемы и решения

### Предупреждение о FFmpeg
Если вы видите сообщение "Couldn't find ffmpeg or avconv", убедитесь, что файлы FFmpeg (.exe и .dll) скопированы в корневую папку проекта или добавлены в системную переменную PATH.

### CUDA недоступен
Если вы видите сообщение "CUDA недоступен. Будет использоваться CPU", проверьте:
- Установлена ли у вас совместимая видеокарта NVIDIA
- Установлены ли драйверы NVIDIA и CUDA Toolkit
- Установлена ли версия PyTorch с поддержкой CUDA

### Медленная транскрибация
- Используйте GPU если возможно
- Попробуйте менее требовательные модели (medium или small)
- Уменьшите значения параметров beam_size и best_of

## Лицензия

[MIT License](LICENSE)

## Благодарности

- [OpenAI Whisper](https://github.com/openai/whisper) - за отличную модель распознавания речи
- [FFmpeg](https://ffmpeg.org/) - за обработку аудиофайлов
- [PyQt](https://www.riverbankcomputing.com/software/pyqt/) - за фреймворк для создания графического интерфейса
