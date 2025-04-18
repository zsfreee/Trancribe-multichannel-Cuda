# Whisper Transcription Tool

Инструмент для транскрибации стерео-аудиофайлов с разделением на каналы для отдельной обработки речи менеджера и клиента. Приложение автоматически разделяет стерео-записи телефонных разговоров на отдельные каналы и создает текстовую транскрипцию с учетом говорящих.

## Особенности

- Автоматическое разделение стерео-записи на два канала (левый - менеджер, правый - клиент)
- Отдельная транскрибация для каждого канала с индивидуальными настройками
- Создание объединенного диалога с указанием времени
- Сохранение результатов в текстовом и CSV форматах
- Удобный графический интерфейс для настройки параметров
- Веб-интерфейс на базе Streamlit для работы через браузер
- Поддержка русского, казахского и английского языков
- Автоматическое определение и использование GPU (CUDA) для ускорения транскрибации
- Возможность остановки процесса транскрибации
- CLI-версия для работы через командную строку или на сервере без GUI
- Скрипт автоматизации для мониторинга директории и обработки новых файлов

## Требования

- Python 3.12+
- FFmpeg (требуется для обработки аудио)
- NVIDIA GPU с CUDA для ускорения (опционально)
- Git https://git-scm.com/downloads/

## Установка

### 1. Клонируйте репозиторий:

```bash
git clone https://github.com/zsfreee/Trancribe-multichannel-Cuda.git
cd Trancribe-multichannel-Cuda
```

### 2. Создайте виртуальное окружение и активируйте его:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Обновите pip:
Windows
```bash
python.exe -m pip install --upgrade pip
```
Linux
```bash
pip install --upgrade pip
```

### 4. Установите PyTorch с поддержкой CUDA (для использования GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```bash
# Для NVIDIA GPU:
pip install -r requirements.txt
```

### 4.1 Установите PyTorch БЕЗ поддержкой CUDA (для использования только на CPU):
```bash
pip install torch==2.6.0+cpu torchvision==0.21.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
```
```bash
pip install -r requirements_cpu.txt
```
### Проверяем режим работы PyTorch
```bash
python -c "import torch; print('CUDA доступен:', torch.cuda.is_available()); print('Устройство:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 5. Установите FFmpeg:

**Windows:**
- Скачайте FFmpeg с официального сайта или с GitHub (выберите версию с "shared" в названии)
- Распакуйте архив
- Скопируйте все файлы .exe и .dll из папки bin в корень проекта

Например, для Windows 64bit - необходимо перенести в корневую папку проекта следующие файлы из папки bin ffmpeg:
```
avcodec-61.dll
avdevice-61.dll
avfilter-10.dll
avformat-61.dll
avutil-59.dll
postproc-58.dll
swresample-5.dll
swscale-8.dll
ffmpeg.exe
ffplay.exe
ffprobe.exe
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
sudo apt install -y ffmpeg libavcodec-extra
```

**Mac:**
```bash
brew install ffmpeg
```

## Использование

### Графический интерфейс (GUI)

Запустите приложение:
```bash
python whisper_app.py
```

В интерфейсе приложения:
1. На вкладке "Файлы" выберите папку с аудиофайлами для обработки и папку для сохранения результатов
2. На вкладке "Параметры" настройте параметры модели Whisper для каждого канала
3. На вкладке "Выполнение" нажмите "Начать транскрибацию" и следите за процессом в журнале

### Веб-интерфейс (Streamlit)

#### Установка Streamlit-версии:

##### Полная установка с поддержкой CUDA (для использования GPU):
```bash
pip install -r requirements_streamlit.txt
```

##### Полная установка без поддержки CUDA (только CPU):
```bash
pip install torch==2.6.0+cpu torchvision==0.21.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_streamlit_cpu.txt
```

##### Минимальная установка (если у вас уже установлены основные зависимости):
```bash
pip install -r streamlit_minimal.txt
```

#### Запуск Streamlit-приложения:
```bash
streamlit run streamlit_app.py
```

Приложение будет доступно по адресу http://localhost:8501

В веб-интерфейсе:
1. На вкладке "📁 Файлы" загрузите аудиофайлы для обработки
2. На вкладке "⚙️ Параметры" настройте параметры модели Whisper для каждого канала
3. На вкладке "▶️ Выполнение" нажмите "🚀 Начать транскрибацию" и следите за процессом
4. После завершения скачайте результаты в виде ZIP-архива

#### Проверка режима работы (CUDA/CPU):
Информация о текущем режиме работы (CUDA GPU или CPU) отображается в заголовке приложения.

#### Развертывание Streamlit-приложения на сервере:

1. **Установка на сервер**:
   ```bash
   # Для GPU-версии:
   pip install -r requirements_streamlit.txt
   
   # Для CPU-версии:
   pip install torch==2.6.0+cpu torchvision==0.21.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements_streamlit_cpu.txt
   ```

2. **Запуск на сервере с доступом из внешней сети**:
   ```bash
   streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
   ```

3. **Настройка постоянной работы через systemd**:
   
   Создайте файл сервиса:
   ```bash
   sudo nano /etc/systemd/system/whisper-streamlit.service
   ```
   
   Содержимое файла:
   ```
   [Unit]
   Description=Whisper Transcription Streamlit Server
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/path/to/Trancribe-multichannel-Cuda
   ExecStart=/path/to/venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```
   
   Активация и запуск:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable whisper-streamlit
   sudo systemctl start whisper-streamlit
   ```

### Запуск GUI на сервере через X11 (для VPS без графического интерфейса)

#### Настройка на стороне сервера:

```bash
# Установка необходимых пакетов
sudo apt install -y xauth x11-apps

# Проверьте, что X11 forwarding разрешен в SSH
sudo nano /etc/ssh/sshd_config
```

Убедитесь, что в файле есть следующие строки и они не закомментированы:
```
X11Forwarding yes
X11DisplayOffset 10
X11UseLocalhost yes
```

Если внесли изменения, перезагрузите SSH сервис:
```bash
sudo systemctl restart ssh
```

#### Настройка на стороне клиента:

**Windows:**
1. Установите X-сервер, например VcXsrv (https://sourceforge.net/projects/vcxsrv/) или Xming (https://sourceforge.net/projects/xming/)
2. Запустите X-сервер
3. При подключении через PuTTY:
   - В настройках соединения перейдите в Connection → SSH → X11
   - Установите галочку "Enable X11 forwarding"
   - В поле "X display location" оставьте значение по умолчанию

**Linux или macOS:**
```bash
ssh -X username@your_server_ip
```

#### Запуск приложения через X11:

```bash
cd ~/Trancribe-multichannel-Cuda
source venv/bin/activate
python whisper_app.py
```

### Командная строка (CLI)

Вы можете использовать CLI-версию приложения для транскрибации без графического интерфейса:

```bash
python cli_transcribe.py --input <директория_с_аудио> --output <директория_для_результатов> [дополнительные_опции]
```

**Обязательные параметры:**
- `--input` или `-i`: Путь к директории с исходными аудиофайлами
- `--output` или `-o`: Путь к директории для сохранения результатов

**Дополнительные параметры:**
- `--model` или `-m`: Размер модели Whisper (`tiny`, `small`, `medium`, `large-v3`, по умолчанию `small`)
- `--left-lang`: Язык для левого канала/менеджера (по умолчанию `Русский`)
- `--right-lang`: Язык для правого канала/клиента (по умолчанию `Русский`)

**Примеры запуска CLI:**

Базовый запуск с параметрами по умолчанию:
```bash
python cli_transcribe.py --input ~/test_audio --output ~/test_output
```

Запуск с легкой моделью для ограниченных ресурсов:
```bash
python cli_transcribe.py --input ~/test_audio --output ~/test_output --model tiny
```

Запуск с автоматическим определением языка:
```bash
python cli_transcribe.py --input ~/test_audio --output ~/test_output --left-lang "Автоматическое определение" --right-lang "Автоматическое определение"
```

### Автоматизация с помощью скрипта мониторинга

Скрипт `watch_and_transcribe.py` позволяет автоматически обрабатывать новые аудиофайлы, появляющиеся в указанной директории:

```bash
./watch_and_transcribe.py [параметры]
```

**Параметры скрипта мониторинга:**
- `--watch-dir`: Директория для мониторинга (по умолчанию `/home/ubuntu/test_audio`)
- `--processed-dir`: Директория для обработанных файлов (по умолчанию `/home/ubuntu/test_audio_complete`)
- `--output-dir`: Директория для результатов (по умолчанию `/home/ubuntu/test_output`)
- `--interval`: Интервал проверки в секундах (по умолчанию `30`)
- `--model`: Размер модели Whisper (по умолчанию `small`)
- `--left-lang`: Язык левого канала (по умолчанию `Русский`)
- `--right-lang`: Язык правого канала (по умолчанию `Русский`)

**Примеры запуска скрипта мониторинга:**

Запуск с параметрами по умолчанию:
```bash
./watch_and_transcribe.py
```

Полная настройка всех параметров:
```bash
./watch_and_transcribe.py --watch-dir /путь/к/аудио --processed-dir /путь/к/обработанным --output-dir /путь/к/результатам --interval 60 --model small --left-lang "Русский" --right-lang "Русский"
```

### Запуск скрипта мониторинга как системного сервиса

Чтобы запустить скрипт мониторинга как системный сервис на Linux:

```bash
# Создайте файл сервиса
sudo nano /etc/systemd/system/transcribe-monitor.service
```

Вставьте следующее содержимое (замените пути при необходимости):
```
[Unit]
Description=Audio Transcription Monitor
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/Trancribe-multichannel-Cuda
ExecStart=/home/ubuntu/Trancribe-multichannel-Cuda/venv/bin/python /home/ubuntu/Trancribe-multichannel-Cuda/watch_and_transcribe.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Активируйте и запустите сервис:
```bash
sudo systemctl daemon-reload
sudo systemctl enable transcribe-monitor
sudo systemctl start transcribe-monitor
```

Для проверки статуса:
```bash
sudo systemctl status transcribe-monitor
```

Для просмотра логов:
```bash
sudo journalctl -u transcribe-monitor -f
```

## Структура выходных файлов

После завершения транскрибации в выбранной папке будут созданы следующие подпапки:

- "Только менеджер" - содержит транскрибацию левого канала
- "Только клиент" - содержит транскрибацию правого канала
- "С таймкодами" - содержит объединенный диалог с временными метками
- "Без таймкодов" - содержит объединенный диалог без временных меток
- "CSV transcribe" - содержит CSV-файлы со всеми типами транскрибаций, включая общий файл all_transcriptions.csv

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
- >0.5: Более разнообразные результаты (может помочь при нечеткой речи)

#### Beam size (1-10)
Количество лучей при поиске лучшей транскрипции:
- 1: Самый быстрый, но менее точный (жадный поиск)
- 5: Рекомендуемое значение (баланс между скоростью и точностью)
- >5: Более точно, но значительно медленнее

#### Patience (0.5-2.0)
Коэффициент терпения при поиске:
- 1.0: Стандартное значение
- <1.0: Быстрее, но менее точно
- >1.0: Медленнее, но потенциально более точно

#### Best of (1, 3, 5, 7, 10)
Количество сэмплов для выбора лучшего результата:
- 1: Самый быстрый, но менее точный
- 5: Рекомендуемое значение (хороший баланс)
- >5: Более точно, но требует больше ресурсов

#### No speech threshold (0.1-1.0)
Порог для определения отсутствия речи:
- 0.1-0.3: Меньше пропусков тихой речи, но больше ложных срабатываний
- 0.6: Рекомендуемое значение
- >0.7: Меньше ложных срабатываний, но больше риск пропустить тихую речи

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

### Для сервера с ограниченными ресурсами (1 CPU, 2GB RAM)
- Модель: tiny
- Язык: Соответствующий язык разговора
- Температура: 0.0
- Beam size: 1
- Patience: 1.0
- Best of: 1
- No speech threshold: 0.6

## Управление моделями Whisper

Модели Whisper хранятся в кэше по пути `~/.cache/whisper/`. Для экономии места вы можете удалить ненужные модели:

```bash
# Просмотр скачанных моделей
ls -lah ~/.cache/whisper/

# Удаление конкретных моделей
rm -f ~/.cache/whisper/medium.pt
rm -f ~/.cache/whisper/large-v3.pt

# Оставить только модель tiny
find ~/.cache/whisper/ -type f -not -name "tiny.pt" -delete
```

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
- Попробуйте менее требовательные модели (tiny или small)
- Уменьшите значения параметров beam_size и best_of

### Ошибка при запуске скрипта мониторинга
Если вы видите ошибку "No such file or directory: 'python'", убедитесь, что в скрипте `watch_and_transcribe.py` используется правильный путь к Python:
```python
# Вместо
cmd = ["python", ...]
# Используйте
cmd = [sys.executable, ...]
```

## Мониторинг использования ресурсов

Для мониторинга использования памяти и процессора в реальном времени используйте:

```bash
watch -n 1 "free -h; echo; echo 'CPU Usage:'; top -bn1 | head -3"
```

## Лицензия

MIT License

## Благодарности

- OpenAI Whisper - за отличную модель распознавания речи
- FFmpeg - за обработку аудиофайлов
- PyQt - за фреймворк для создания графического интерфейса
