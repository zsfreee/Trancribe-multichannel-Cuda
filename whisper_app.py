import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QComboBox, QWidget, QTabWidget,
                            QProgressBar, QLineEdit, QGroupBox, QGridLayout, QMessageBox,
                            QSpinBox, QDoubleSpinBox, QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import whisper
from pydub import AudioSegment
import glob
import time
import pandas as pd
import torch

class TranscriptionWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    status = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, input_dir, output_dir, params):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.params = params
        self.running = True
        
    def run(self):
        try:
            # Проверка доступности GPU
            self.log.emit(f"CUDA доступен: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self.log.emit(f"Используемое устройство: {torch.cuda.get_device_name(0)}")
                device = torch.device("cuda")
            else:
                self.log.emit("GPU не доступен, используется CPU")
                device = torch.device("cpu")
            
            # Указываем директорию кэша явно
            from pathlib import Path
            cache_dir = Path.home() / ".cache" / "whisper"
            os.makedirs(str(cache_dir), exist_ok=True)
            
            # Загрузка модели
            self.log.emit(f"Загрузка модели {self.params['model_size']}...")
            model = whisper.load_model(self.params['model_size'], download_root=str(cache_dir)).to(device)
            self.log.emit("Модель успешно загружена на " + ("GPU" if torch.cuda.is_available() else "CPU"))
            
            # Поиск аудиофайлов
            audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(glob.glob(os.path.join(self.input_dir, f"*{ext}")))
            
            total_files = len(audio_files)
            if total_files == 0:
                self.log.emit(f"Аудиофайлы не найдены в папке {self.input_dir}")
                self.status.emit("Ошибка: файлы не найдены")
                return
            
            self.log.emit(f"Найдено аудиофайлов: {total_files}")
            
            # Сопоставление языка с кодом для Whisper
            language_map = {
                "Автоматическое определение": None,
                "Русский": "ru",
                "Казахский": "kk",
                "Английский": "en"
            }
            
            # Создание структуры папок
            subdirs = {
                "manager": os.path.join(self.output_dir, "Только менеджер"),
                "client": os.path.join(self.output_dir, "Только клиент"),
                "with_time": os.path.join(self.output_dir, "С таймкодами"),
                "no_time": os.path.join(self.output_dir, "Без таймкодов")
            }
            
            for folder_path in subdirs.values():
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            
            csv_dir = os.path.join(self.output_dir, "CSV transcribe")
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            
            # Обработка файлов
            successful = 0
            failed = 0
            
            for i, audio_file in enumerate(audio_files):
                if not self.running:
                    self.log.emit("Транскрибация прервана пользователем.")
                    break
                
                file_name = os.path.basename(audio_file).split('.')[0]
                self.log.emit(f"\nОбработка файла {i+1} из {total_files}: {file_name}")
                self.status.emit(f"Обработка: {i+1}/{total_files}")
                
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
                    self.log.emit("Транскрибация канала менеджера...")
                    
                    transcribe_options_manager = {
                        "temperature": self.params['left_temperature'],
                        "beam_size": self.params['left_beam_size'],
                        "patience": self.params['left_patience'],
                        "best_of": int(self.params['left_best_of']),
                        "no_speech_threshold": self.params['left_no_speech']
                    }
                    
                    if language_map[self.params['left_language']]:
                        transcribe_options_manager["language"] = language_map[self.params['left_language']]
                    
                    result_manager = model.transcribe(manager_audio_path, **transcribe_options_manager)
                    
                    # Транскрибация канала клиента
                    self.log.emit("Транскрибация канала клиента...")
                    
                    transcribe_options_client = {
                        "temperature": self.params['right_temperature'],
                        "beam_size": self.params['right_beam_size'],
                        "patience": self.params['right_patience'],
                        "best_of": int(self.params['right_best_of']),
                        "no_speech_threshold": self.params['right_no_speech']
                    }
                    
                    if language_map[self.params['right_language']]:
                        transcribe_options_client["language"] = language_map[self.params['right_language']]
                    
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
                    self.log.emit(f"✅ Файл обработан успешно: {file_name}")
                    
                except Exception as e:
                    failed += 1
                    self.log.emit(f"❌ Ошибка при обработке файла {file_name}: {str(e)}")
                
                # Обновление прогресса
                progress_value = int(((i + 1) / total_files) * 100)
                self.progress.emit(progress_value)
            
            # Создание CSV файлов
            self.log.emit("\nСоздание CSV файлов...")
            
            # Функция для чтения содержимого текстового файла
            def read_file_content(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        return file.read()
                except Exception as e:
                    self.log.emit(f"Ошибка при чтении файла {file_path}: {str(e)}")
                    return ""
            
            # Создание отдельных CSV для каждой папки
            for folder_name, folder_path in subdirs.items():
                txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
                
                if not txt_files:
                    self.log.emit(f"В папке {folder_path} нет текстовых файлов.")
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
                self.log.emit(f"Создан CSV файл для папки {folder_label}")
            
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
            
            self.log.emit(f"Создан общий CSV файл: {combined_csv_path}")
            
            # Итоги
            self.log.emit("\n===== Итоги обработки =====")
            self.log.emit(f"Всего файлов: {total_files}")
            self.log.emit(f"Успешно обработано: {successful}")
            self.log.emit(f"С ошибками: {failed}")
            self.status.emit("Транскрибация завершена")
            
        except Exception as e:
            self.log.emit(f"Ошибка в процессе транскрибации: {str(e)}")
            self.status.emit("Ошибка")
        
        self.finished_signal.emit()
    
    def stop(self):
        self.running = False


class WhisperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Whisper Transcription Tool")
        self.setGeometry(100, 100, 900, 600)
        
        # Устанавливаем иконку приложения
        icon_path = "icon.ico"  # Убедитесь, что файл icon.ico находится в той же папке, что и скрипт
        if os.path.exists(icon_path):
            from PyQt6.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))

        # Основной виджет и компоновка
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Создаем вкладки
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Вкладка "Файлы"
        files_tab = QWidget()
        files_layout = QVBoxLayout(files_tab)
        
        # Выбор входной папки
        input_group = QGroupBox("Входная папка с аудиофайлами")
        input_layout = QHBoxLayout()
        
        self.input_path = QLineEdit()
        self.input_path.setReadOnly(True)
        input_layout.addWidget(self.input_path)
        
        input_button = QPushButton("Обзор...")
        input_button.clicked.connect(self.select_input)
        input_layout.addWidget(input_button)
        
        input_group.setLayout(input_layout)
        files_layout.addWidget(input_group)
        
        # Выбор выходной папки
        output_group = QGroupBox("Выходная папка для результатов")
        output_layout = QHBoxLayout()
        
        self.output_path = QLineEdit()
        self.output_path.setReadOnly(True)
        output_layout.addWidget(self.output_path)
        
        output_button = QPushButton("Обзор...")
        output_button.clicked.connect(self.select_output)
        output_layout.addWidget(output_button)
        
        output_group.setLayout(output_layout)
        files_layout.addWidget(output_group)
        
        # Добавляем layout файлов в вкладку
        files_tab.setLayout(files_layout)
        tabs.addTab(files_tab, "Файлы")
        
        # Вкладка "Параметры"
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)
        
        # Выбор модели
        model_group = QGroupBox("Модель Whisper")
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Размер модели:"))
        self.model_size = QComboBox()
        self.model_size.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model_size.setCurrentText("large-v3")  # По умолчанию
        model_layout.addWidget(self.model_size)
        model_group.setLayout(model_layout)
        params_layout.addWidget(model_group)
        
        # Группа параметров для левого канала (Менеджер)
        left_group = QGroupBox("Параметры для левого канала (Менеджер)")
        left_layout = QGridLayout()
        
        # Выбор языка для левого канала
        left_layout.addWidget(QLabel("Язык:"), 0, 0)
        self.left_language = QComboBox()
        self.left_language.addItems(["Автоматическое определение", "Русский", "Казахский", "Английский"])
        self.left_language.setCurrentText("Русский")
        left_layout.addWidget(self.left_language, 0, 1)
        
        # Температура для левого канала
        left_layout.addWidget(QLabel("Температура (0-1):"), 1, 0)
        self.left_temperature = QDoubleSpinBox()
        self.left_temperature.setRange(0.0, 1.0)
        self.left_temperature.setSingleStep(0.1)
        self.left_temperature.setValue(0.2)
        left_layout.addWidget(self.left_temperature, 1, 1)
        
        # Beam size для левого канала
        left_layout.addWidget(QLabel("Beam size (1-10):"), 2, 0)
        self.left_beam_size = QSpinBox()
        self.left_beam_size.setRange(1, 10)
        self.left_beam_size.setValue(5)
        left_layout.addWidget(self.left_beam_size, 2, 1)
        
        # Patience для левого канала
        left_layout.addWidget(QLabel("Patience (0.5-2.0):"), 3, 0)
        self.left_patience = QDoubleSpinBox()
        self.left_patience.setRange(0.5, 2.0)
        self.left_patience.setSingleStep(0.1)
        self.left_patience.setValue(1.0)
        left_layout.addWidget(self.left_patience, 3, 1)
        
        # Best of для левого канала
        left_layout.addWidget(QLabel("Best of:"), 4, 0)
        self.left_best_of = QComboBox()
        self.left_best_of.addItems(["1", "3", "5", "7", "10"])
        self.left_best_of.setCurrentText("5")
        left_layout.addWidget(self.left_best_of, 4, 1)
        
        # No speech threshold для левого канала
        left_layout.addWidget(QLabel("No speech threshold (0.1-1.0):"), 5, 0)
        self.left_no_speech = QDoubleSpinBox()
        self.left_no_speech.setRange(0.1, 1.0)
        self.left_no_speech.setSingleStep(0.1)
        self.left_no_speech.setValue(0.6)
        left_layout.addWidget(self.left_no_speech, 5, 1)
        
        left_group.setLayout(left_layout)
        params_layout.addWidget(left_group)
        
        # Группа параметров для правого канала (Клиент)
        right_group = QGroupBox("Параметры для правого канала (Клиент)")
        right_layout = QGridLayout()
        
        # Выбор языка для правого канала
        right_layout.addWidget(QLabel("Язык:"), 0, 0)
        self.right_language = QComboBox()
        self.right_language.addItems(["Автоматическое определение", "Русский", "Казахский", "Английский"])
        self.right_language.setCurrentText("Русский")
        right_layout.addWidget(self.right_language, 0, 1)
        
        # Температура для правого канала
        right_layout.addWidget(QLabel("Температура (0-1):"), 1, 0)
        self.right_temperature = QDoubleSpinBox()
        self.right_temperature.setRange(0.0, 1.0)
        self.right_temperature.setSingleStep(0.1)
        self.right_temperature.setValue(0.2)
        right_layout.addWidget(self.right_temperature, 1, 1)
        
        # Beam size для правого канала
        right_layout.addWidget(QLabel("Beam size (1-10):"), 2, 0)
        self.right_beam_size = QSpinBox()
        self.right_beam_size.setRange(1, 10)
        self.right_beam_size.setValue(5)
        right_layout.addWidget(self.right_beam_size, 2, 1)
        
        # Patience для правого канала
        right_layout.addWidget(QLabel("Patience (0.5-2.0):"), 3, 0)
        self.right_patience = QDoubleSpinBox()
        self.right_patience.setRange(0.5, 2.0)
        self.right_patience.setSingleStep(0.1)
        self.right_patience.setValue(1.0)
        right_layout.addWidget(self.right_patience, 3, 1)
        
        # Best of для правого канала
        right_layout.addWidget(QLabel("Best of:"), 4, 0)
        self.right_best_of = QComboBox()
        self.right_best_of.addItems(["1", "3", "5", "7", "10"])
        self.right_best_of.setCurrentText("5")
        right_layout.addWidget(self.right_best_of, 4, 1)
        
        # No speech threshold для правого канала
        right_layout.addWidget(QLabel("No speech threshold (0.1-1.0):"), 5, 0)
        self.right_no_speech = QDoubleSpinBox()
        self.right_no_speech.setRange(0.1, 1.0)
        self.right_no_speech.setSingleStep(0.1)
        self.right_no_speech.setValue(0.6)
        right_layout.addWidget(self.right_no_speech, 5, 1)
        
        right_group.setLayout(right_layout)
        params_layout.addWidget(right_group)
        
        params_tab.setLayout(params_layout)
        tabs.addTab(params_tab, "Параметры")
        
        # Вкладка "Выполнение"
        run_tab = QWidget()
        run_layout = QVBoxLayout(run_tab)
        
        # Журнал выполнения
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        run_layout.addWidget(QLabel("Журнал выполнения:"))
        run_layout.addWidget(self.log_text)
        
        # Статус
        self.status_label = QLabel("Готов к работе")
        run_layout.addWidget(self.status_label)
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        run_layout.addWidget(QLabel("Прогресс:"))
        run_layout.addWidget(self.progress_bar)
        
        # Кнопки запуска и остановки
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Начать транскрибацию")
        self.start_button.clicked.connect(self.start_transcription)
        buttons_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Остановить транскрибацию")
        self.stop_button.clicked.connect(self.stop_transcription)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        run_layout.addLayout(buttons_layout)
        
        run_tab.setLayout(run_layout)
        tabs.addTab(run_tab, "Выполнение")
        
        # Проверка CUDA при запуске
        self.check_cuda()
    
    def check_cuda(self):
        """Проверка доступности CUDA"""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                self.log_text.append(f"CUDA доступен. Обнаружено устройство: {device_name}")
                self.log_text.append(f"Версия CUDA: {cuda_version}")
            else:
                self.log_text.append("CUDA недоступен. Будет использоваться CPU.")
                self.log_text.append("Для ускорения работы рекомендуется установить PyTorch с поддержкой CUDA.")
        except Exception as e:
            self.log_text.append(f"Ошибка при проверке CUDA: {str(e)}")
    
    def select_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с аудиофайлами")
        if folder:
            self.input_path.setText(folder)
    
    def select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения результатов")
        if folder:
            self.output_path.setText(folder)
    
    def start_transcription(self):
        if not self.input_path.text():
            QMessageBox.warning(self, "Предупреждение", "Выберите папку с аудиофайлами!")
            return
        
        if not self.output_path.text():
            QMessageBox.warning(self, "Предупреждение", "Выберите папку для сохранения результатов!")
            return
        
        # Сбор параметров для передачи обработчику
        params = {
            'model_size': self.model_size.currentText(),
            'left_language': self.left_language.currentText(),
            'left_temperature': self.left_temperature.value(),
            'left_beam_size': self.left_beam_size.value(),
            'left_patience': self.left_patience.value(),
            'left_best_of': self.left_best_of.currentText(),
            'left_no_speech': self.left_no_speech.value(),
            'right_language': self.right_language.currentText(),
            'right_temperature': self.right_temperature.value(),
            'right_beam_size': self.right_beam_size.value(),
            'right_patience': self.right_patience.value(),
            'right_best_of': self.right_best_of.currentText(),
            'right_no_speech': self.right_no_speech.value()
        }
        
        # Очистка журнала и сброс прогресса
        self.log_text.clear()
        self.progress_bar.setValue(0)
        
        # Создание и запуск потока обработки
        self.worker = TranscriptionWorker(
            self.input_path.text(),
            self.output_path.text(),
            params
        )
        
        # Подключение сигналов
        self.worker.log.connect(self.update_log)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished_signal.connect(self.transcription_finished)
        
        # Запуск обработки
        self.worker.start()
        
        # Обновление состояния интерфейса
        self.status_label.setText("Выполняется транскрибация...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def stop_transcription(self):
        """Остановка процесса транскрибации"""
        if hasattr(self, 'worker') and self.worker.isRunning():
            reply = QMessageBox.question(
                self, 
                "Подтверждение", 
                "Вы уверены, что хотите остановить транскрибацию?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.log_text.append("Остановка транскрибации...")
                self.worker.stop()
                self.stop_button.setEnabled(False)
    
    def update_log(self, text):
        """Обновление журнала событий"""
        self.log_text.append(text)
        # Прокрутка до конца
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def transcription_finished(self):
        """Обработка завершения транскрибации"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        QMessageBox.information(self, "Информация", "Транскрибация завершена")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WhisperApp()
    window.show()
    sys.exit(app.exec())
