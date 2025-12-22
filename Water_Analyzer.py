
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# PyQt6 импорты
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QFileDialog, QTextEdit, QMessageBox, QGroupBox,
                             QSplitter, QTabWidget, QFormLayout,
                             QProgressBar, QGridLayout, QDialog, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor

# Машинное обучение - для графиков прогнозирования
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Визуализация
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats


class PlotWindow(QDialog):
    """Класс для окна с графиком"""

    def __init__(self, title="График", parent=None):
        super().__init__(parent)
        self.title = title
        self.figure = None
        self.canvas = None
        self.setup_ui()

    def setup_ui(self):
        """Настройка интерфейса окна"""
        self.setWindowTitle(self.title)
        self.setGeometry(200, 100, 1000, 700)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Создание фигуры и холста
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)

        # Добавление панели инструментов
        toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

    def plot_data(self, plot_function, *args, **kwargs):
        """Построение графика в окне"""
        try:
            # Очищаем предыдущий график
            self.figure.clear()

            # Вызываем функцию построения графика
            ax = self.figure.add_subplot(111)
            plot_function(ax, *args, **kwargs)

            # Автоматическая подстройка layout
            self.figure.tight_layout()

            # Обновление холста
            self.canvas.draw()

        except Exception as e:
            self.show_error_message(str(e))

    def show_error_message(self, message):
        """Показ сообщения об ошибке на графике"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, f'Ошибка:\n{message}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='red', wrap=True)
        ax.set_axis_off()
        self.canvas.draw()


class TextWindow(QDialog):
    """Класс для окна с текстовой информацией"""

    def __init__(self, title="Информация", parent=None):
        super().__init__(parent)
        self.title = title
        self.setup_ui()

    def setup_ui(self):
        """Настройка интерфейса окна"""
        self.setWindowTitle(self.title)
        self.setGeometry(300, 150, 900, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Создание текстового поля с прокруткой
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        self.text_widget = QTextEdit()
        self.text_widget.setReadOnly(True)
        font = QFont("Courier New", 10)
        self.text_widget.setFont(font)

        scroll_area.setWidget(self.text_widget)

        # Кнопка закрытия
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.close)
        btn_close.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-weight: bold;
                background-color: #607D8B;
                color: white;
                border-radius: 5px;
            }
        """)

        layout.addWidget(scroll_area)
        layout.addWidget(btn_close)

    def set_text(self, text):
        """Установка текста в окно"""
        self.text_widget.setText(text)


class StatisticalAnalyzer(QMainWindow):
    """Главное окно приложения для анализа статистических данных"""

    def __init__(self):
        super().__init__()
        self.df = None
        self.current_file = None
        self.plot_windows = []  # Список открытых окон с графиками
        self.text_windows = []  # Список открытых текстовых окон
        self.init_ui()

    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("Анализатор статистики водопотребления")
        self.setGeometry(100, 100, 1400, 900)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Главный layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Панель управления
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Панель кнопок анализа (статистика + корреляция)
        analysis_buttons_panel = self.create_analysis_buttons_panel()
        main_layout.addWidget(analysis_buttons_panel)

        # Панель кнопок графиков (ВСЕ 8 КНОПОК!)
        plot_buttons_panel = self.create_plot_buttons_panel()
        main_layout.addWidget(plot_buttons_panel)

        # Разделитель с вкладками
        splitter = QSplitter(Qt.Orientation.Vertical)

        # ТОЛЬКО ОДНА ВКЛАДКА - Консоль
        self.tab_widget = QTabWidget()
        self.tab_console = QWidget()  # Консоль

        self.setup_console_tab()

        self.tab_widget.addTab(self.tab_console, "📝 Консоль")

        splitter.addWidget(self.tab_widget)

        # Увеличиваем место для консоли
        splitter.setSizes([150, 750])
        main_layout.addWidget(splitter)

        # Статус бар
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_control_panel(self):
        """Создание панели управления"""
        panel = QGroupBox("Управление")
        layout = QGridLayout()

        # Кнопка загрузки файла
        self.btn_load = QPushButton("📂 Загрузить Excel файл")
        self.btn_load.clicked.connect(self.load_file)
        self.btn_load.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-weight: bold;
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.btn_load, 0, 0, 1, 2)

        # Информация о файле
        self.file_label = QLabel("Файл не загружен")
        self.file_label.setStyleSheet("color: gray; font-style: italic; padding: 5px;")
        layout.addWidget(self.file_label, 1, 0, 1, 2)

        # Выбор столбцов для анализа
        layout.addWidget(QLabel("Столбец для статистики:"), 2, 0)
        self.combo_single = QComboBox()
        layout.addWidget(self.combo_single, 2, 1)

        layout.addWidget(QLabel("Первый столбец для корреляции:"), 3, 0)
        self.combo_corr1 = QComboBox()
        layout.addWidget(self.combo_corr1, 3, 1)

        layout.addWidget(QLabel("Второй столбец для корреляции:"), 4, 0)
        self.combo_corr2 = QComboBox()
        layout.addWidget(self.combo_corr2, 4, 1)

        # Кнопка расчета
        self.btn_calculate = QPushButton("🧮 Рассчитать данные")
        self.btn_calculate.clicked.connect(self.perform_calculation)
        self.btn_calculate.setEnabled(False)
        self.btn_calculate.setStyleSheet("""
            QPushButton {
                padding: 10px;
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.btn_calculate, 5, 0, 1, 2)

        panel.setLayout(layout)
        return panel

    def create_analysis_buttons_panel(self):
        """Создание панели кнопок для анализа (статистика + корреляция)"""
        panel = QGroupBox("Анализ данных")
        layout = QGridLayout()

        # Кнопки для анализа
        self.btn_statistics = self.create_analysis_button("📊 Статистический анализ",
                                                          self.open_statistics_window, "#4CAF50")
        self.btn_correlation = self.create_analysis_button("🔗 Корреляционный анализ",
                                                           self.open_correlation_window, "#9C27B0")

        # Расположение кнопок
        layout.addWidget(self.btn_statistics, 0, 0)
        layout.addWidget(self.btn_correlation, 0, 1)

        panel.setLayout(layout)
        return panel

    def create_plot_buttons_panel(self):
        """Создание панели кнопок для графиков (ВСЕ 8 КНОПОК!)"""
        panel = QGroupBox("Графики")
        layout = QGridLayout()

        # Кнопки для основных графиков (5 штук)
        self.btn_density = self.create_plot_button("📊 Плотность распределения",
                                                   self.open_density_plot, "#2196F3")
        self.btn_histogram = self.create_plot_button("📈 Гистограмма",
                                                     self.open_histogram_plot, "#4CAF50")
        self.btn_box_iqr = self.create_plot_button("📦 Box Plot (IQR)",
                                                   self.open_boxplot_iqr, "#FF9800")
        self.btn_box_std = self.create_plot_button("📊 Box Plot (Mean ± Std)",
                                                   self.open_boxplot_std, "#9C27B0")
        self.btn_scatter = self.create_plot_button("🔵 Scatter Plot",
                                                   self.open_scatter_plot, "#E91E63")

        # Кнопки для прогнозов (3 штуки) - фиксированные параметры
        self.btn_lin_reg = self.create_plot_button("📉 Linear Regression",
                                                   self.open_linear_regression_plot, "#607D8B")
        self.btn_rf = self.create_plot_button("🌲 Random Forest",
                                              self.open_random_forest_plot, "#795548")
        self.btn_dt = self.create_plot_button("🌳 Decision Tree",
                                              self.open_decision_tree_plot, "#009688")

        # Расположение кнопок в сетке 4x2
        layout.addWidget(self.btn_density, 0, 0)
        layout.addWidget(self.btn_histogram, 0, 1)
        layout.addWidget(self.btn_box_iqr, 1, 0)
        layout.addWidget(self.btn_box_std, 1, 1)
        layout.addWidget(self.btn_scatter, 2, 0)
        layout.addWidget(self.btn_lin_reg, 2, 1)
        layout.addWidget(self.btn_rf, 3, 0)
        layout.addWidget(self.btn_dt, 3, 1)

        panel.setLayout(layout)
        return panel

    def create_analysis_button(self, text, slot, color):
        """Создание стилизованной кнопки для анализа"""
        button = QPushButton(text)
        button.clicked.connect(slot)
        button.setEnabled(False)

        button.setStyleSheet(f"""
            QPushButton {{
                padding: 12px;
                font-weight: bold;
                background-color: {color};
                color: white;
                border-radius: 5px;
                margin: 2px;
                font-size: 12px;
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
            QPushButton:hover:enabled {{
                background-color: {color};
                opacity: 0.8;
            }}
        """)

        return button

    def create_plot_button(self, text, slot, color):
        """Создание стилизованной кнопки для графика"""
        button = QPushButton(text)
        button.clicked.connect(slot)
        button.setEnabled(False)

        button.setStyleSheet(f"""
            QPushButton {{
                padding: 10px;
                font-weight: bold;
                background-color: {color};
                color: white;
                border-radius: 5px;
                margin: 2px;
                font-size: 11px;
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
            QPushButton:hover:enabled {{
                background-color: {color};
                opacity: 0.9;
            }}
        """)

        return button

    def setup_console_tab(self):
        """Настройка вкладки консоли"""
        layout = QVBoxLayout()

        # Консоль вывода
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        font = QFont("Consolas", 9)
        self.console.setFont(font)

        layout.addWidget(QLabel("📝 КОНСОЛЬ ВЫВОДА", self))
        layout.addWidget(self.console)

        self.tab_console.setLayout(layout)

    def load_file(self):
        """Загрузка Excel файла"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите Excel файл", "",
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )

        if file_name:
            try:
                self.progress_bar.setValue(30)
                QApplication.processEvents()

                # Чтение файла
                self.df = pd.read_excel(file_name)

                # Проверка формата
                if len(self.df.columns) != 3:
                    raise ValueError("Файл должен содержать ровно 3 столбца")

                # Проверка имен столбцов
                expected_columns = ['Время', 'Активность', 'Объем воды (л)']
                if list(self.df.columns) != expected_columns:
                    self.df.columns = expected_columns

                # Преобразование типов
                self.df['Время'] = pd.to_datetime(self.df['Время'])
                self.df['Объем воды (л)'] = pd.to_numeric(self.df['Объем воды (л)'], errors='coerce')

                # Заполнение пропущенных значений
                self.df['Объем воды (л)'].fillna(self.df['Объем воды (л)'].mean(), inplace=True)

                self.current_file = file_name
                self.file_label.setText(f"Загружен: {file_name.split('/')[-1]}")

                # Обновление комбобоксов
                self.update_comboboxes()

                # Активация кнопок
                self.enable_all_buttons(True)
                self.btn_calculate.setEnabled(True)

                self.log_message(f"✅ Файл успешно загружен: {file_name}")
                self.log_message(f"📊 Количество строк: {len(self.df)}")
                self.log_message(f"🗂️ Столбцы: {list(self.df.columns)}")
                self.log_message("➡️ Выберите столбцы для анализа и нажмите 'Рассчитать данные'")

                self.progress_bar.setValue(100)
                QTimer.singleShot(1000, lambda: self.progress_bar.setValue(0))

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
                self.progress_bar.setValue(0)

    def update_comboboxes(self):
        """Обновление выпадающих списков"""
        columns = list(self.df.columns)

        for combo in [self.combo_single, self.combo_corr1, self.combo_corr2]:
            combo.clear()
            combo.addItems(columns)

        # Установка значений по умолчанию
        if 'Объем воды (л)' in columns:
            self.combo_single.setCurrentText('Объем воды (л)')

        if 'Время' in columns and 'Объем воды (л)' in columns:
            self.combo_corr1.setCurrentText('Объем воды (л)')
            self.combo_corr2.setCurrentText('Время')

    def enable_all_buttons(self, enabled):
        """Активация/деактивация всех кнопок"""
        # Кнопки анализа
        self.btn_statistics.setEnabled(enabled)
        self.btn_correlation.setEnabled(enabled)

        # Кнопки графиков
        for btn in [self.btn_density, self.btn_histogram, self.btn_box_iqr,
                    self.btn_box_std, self.btn_scatter, self.btn_lin_reg,
                    self.btn_rf, self.btn_dt]:
            btn.setEnabled(enabled)

    def perform_calculation(self):
        """Выполнение расчетов"""
        try:
            self.progress_bar.setValue(20)

            # Получение выбранных столбцов
            self.single_col = self.combo_single.currentText()
            self.corr_col1 = self.combo_corr1.currentText()
            self.corr_col2 = self.combo_corr2.currentText()

            # Проверка числовых столбцов
            if self.single_col == 'Активность' or self.single_col == 'Время':
                QMessageBox.warning(self, "Предупреждение",
                                    "Для статистического анализа выберите числовой столбец")
                return

            # Сохраняем результаты расчетов
            self.stats_result = self.calculate_statistics(self.single_col)
            self.corr_result = self.calculate_correlation(self.corr_col1, self.corr_col2)

            self.log_message("✅ Расчеты завершены успешно!")
            self.log_message(f"📈 Статистика для: {self.single_col}")
            self.log_message(f"🔗 Корреляция между: {self.corr_col1} и {self.corr_col2}")
            self.log_message("➡️ Нажмите кнопки анализа для просмотра результатов")

            self.progress_bar.setValue(100)
            QTimer.singleShot(1000, lambda: self.progress_bar.setValue(0))

        except Exception as e:
            QMessageBox.critical(self, "Ошибка расчета", str(e))
            self.progress_bar.setValue(0)

    def open_statistics_window(self):
        """Открытие окна со статистическим анализом"""
        if not hasattr(self, 'stats_result'):
            QMessageBox.warning(self, "Предупреждение",
                                "Сначала выполните расчет данных")
            return

        window = TextWindow(f"Статистический анализ: {self.single_col}", self)
        window.set_text(self.format_statistics_text())
        window.show()
        self.text_windows.append(window)

    def open_correlation_window(self):
        """Открытие окна с корреляционным анализом"""
        if not hasattr(self, 'corr_result'):
            QMessageBox.warning(self, "Предупреждение",
                                "Сначала выполните расчет данных")
            return

        window = TextWindow(f"Корреляционный анализ: {self.corr_col1} vs {self.corr_col2}", self)
        window.set_text(self.format_correlation_text())
        window.show()
        self.text_windows.append(window)

    def calculate_statistics(self, column):
        """Расчет статистических показателей"""
        try:
            if column not in self.df.columns:
                raise ValueError(f"Столбец '{column}' не найден в данных")

            data = self.df[column].dropna()

            if len(data) == 0:
                raise ValueError(f"Столбец '{column}' не содержит данных")

            # Проверка, что данные числовые
            if not pd.api.types.is_numeric_dtype(data):
                raise ValueError(f"Столбец '{column}' не является числовым.")

            # Конвертация в float
            data_numeric = pd.to_numeric(data, errors='coerce').dropna()

            if len(data_numeric) == 0:
                raise ValueError(f"Не удалось преобразовать столбец '{column}' в числовой формат")

            data = data_numeric

            # Основные статистики
            stats = {
                'Количество элементов': int(len(data)),
                'Медиана': float(data.median()),
                'Среднее значение': float(data.mean()),
                'Минимальное значение': float(data.min()),
                'Максимальное значение': float(data.max()),
                'Сумма': float(data.sum()),
                'Диапазон': float(data.max() - data.min()),
            }

            # Мода
            mode_values = data.mode()
            if not mode_values.empty:
                stats['Мода'] = float(mode_values.iloc[0])
                if len(mode_values) > 1:
                    stats['Дополнительные моды'] = [float(x) for x in mode_values.iloc[1:].tolist()]

            # Квантили
            q1 = float(data.quantile(0.25))
            q2 = float(data.quantile(0.50))
            q3 = float(data.quantile(0.75))
            iqr = q3 - q1

            stats.update({
                'Q1 (0.25)': q1,
                'Q2 (медиана)': q2,
                'Q3 (0.75)': q3,
                'IQR': iqr,
                'Нижняя граница выбросов (Q1 - 1.5*IQR)': float(q1 - 1.5 * iqr),
                'Верхняя граница выбросов (Q3 + 1.5*IQR)': float(q3 + 1.5 * iqr)
            })

            # Абсолютные отклонения
            stats['Среднее абсолютное отклонение (MAD)'] = float((data - data.mean()).abs().mean())
            stats['Медианное абсолютное отклонение'] = float((data - data.median()).abs().median())

            # Дисперсия и стандартное отклонение
            stats['Дисперсия'] = float(data.var(ddof=1))
            stats['Стандартное отклонение'] = float(data.std(ddof=1))

            # Асимметрия и эксцесс
            if len(data) > 2:
                stats['Асимметрия (Skewness)'] = float(data.skew())
                stats['Эксцесс (Kurtosis)'] = float(data.kurtosis())

                # Интерпретация асимметрии
                skew = stats['Асимметрия (Skewness)']
                if abs(skew) < 0.5:
                    skew_interpretation = "Симметричное распределение"
                elif 0.5 <= abs(skew) < 1:
                    skew_interpretation = "Умеренная асимметрия"
                else:
                    skew_interpretation = "Сильная асимметрия"
                stats['Интерпретация асимметрии'] = skew_interpretation

            # Коэффициент вариации
            if stats['Среднее значение'] != 0:
                cv = (stats['Стандартное отклонение'] / stats['Среднее значение']) * 100
                stats['Коэффициент вариации (%)'] = float(cv)

            return stats

        except Exception as e:
            raise ValueError(f"Ошибка при расчете статистики: {str(e)}")

    def format_statistics_text(self):
        """Форматирование текста для статистического анализа"""
        text = "═" * 80 + "\n"
        text += "СТАТИСТИЧЕСКИЙ АНАЛИЗ\n"
        text += f"Столбец: {self.single_col}\n"
        text += f"Количество записей: {len(self.df)}\n"
        text += "═" * 80 + "\n\n"

        text += "📊 ОСНОВНЫЕ СТАТИСТИКИ:\n"
        text += "─" * 40 + "\n"

        basic_stats = ['Количество элементов', 'Минимальное значение', 'Максимальное значение',
                       'Среднее значение', 'Медиана', 'Мода', 'Сумма', 'Диапазон']

        for key in basic_stats:
            if key in self.stats_result:
                value = self.stats_result[key]
                if isinstance(value, float):
                    text += f"{key:<45}: {value:>15.4f}\n"
                elif isinstance(value, list):
                    text += f"{key:<45}: {', '.join([f'{v:.4f}' for v in value])}\n"
                else:
                    text += f"{key:<45}: {value:>15}\n"

        text += "\n📈 КВАРТИЛИ И ВЫБРОСЫ:\n"
        text += "─" * 40 + "\n"

        quartile_stats = ['Q1 (0.25)', 'Q2 (медиана)', 'Q3 (0.75)', 'IQR',
                          'Нижняя граница выбросов (Q1 - 1.5*IQR)',
                          'Верхняя граница выбросов (Q3 + 1.5*IQR)']

        for key in quartile_stats:
            if key in self.stats_result:
                value = self.stats_result[key]
                if isinstance(value, float):
                    text += f"{key:<45}: {value:>15.4f}\n"

        text += "\n📉 МЕРЫ РАССЕЯНИЯ:\n"
        text += "─" * 40 + "\n"

        dispersion_stats = ['Среднее абсолютное отклонение (MAD)',
                            'Медианное абсолютное отклонение',
                            'Дисперсия', 'Стандартное отклонение']

        if 'Коэффициент вариации (%)' in self.stats_result:
            dispersion_stats.append('Коэффициент вариации (%)')

        for key in dispersion_stats:
            if key in self.stats_result:
                value = self.stats_result[key]
                if isinstance(value, float):
                    if 'Коэффициент вариации' in key:
                        text += f"{key:<45}: {value:>15.2f}%\n"
                    else:
                        text += f"{key:<45}: {value:>15.4f}\n"

        text += "\n📊 ФОРМА РАСПРЕДЕЛЕНИЯ:\n"
        text += "─" * 40 + "\n"

        shape_stats = ['Асимметрия (Skewness)', 'Эксцесс (Kurtosis)',
                       'Интерпретация асимметрии']

        for key in shape_stats:
            if key in self.stats_result:
                value = self.stats_result[key]
                if isinstance(value, float):
                    text += f"{key:<45}: {value:>15.4f}\n"
                else:
                    text += f"{key:<45}: {value:>15}\n"

        # Интерпретация эксцесса
        if 'Эксцесс (Kurtosis)' in self.stats_result:
            kurtosis = self.stats_result['Эксцесс (Kurtosis)']
            if kurtosis > 0:
                kurt_interpretation = "Островершинное (лептокуртическое)"
            elif kurtosis < 0:
                kurt_interpretation = "Плосковершинное (платикуртическое)"
            else:
                kurt_interpretation = "Нормальное (мезокуртическое)"
            text += f"{'Интерпретация эксцесса':<45}: {kurt_interpretation:>15}\n"

        text += "\n" + "═" * 80 + "\n"
        text += f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += "═" * 80 + "\n"

        return text

    def calculate_correlation(self, col1, col2):
        """Расчет корреляции"""
        try:
            if not col1 or not col2:
                return {'Ошибка': 'Не выбраны столбцы для анализа'}

            # Преобразование времени в числовой формат
            if col1 == 'Время':
                data1 = pd.to_numeric(pd.to_datetime(self.df[col1]))
            else:
                data1 = self.df[col1]

            if col2 == 'Время':
                data2 = pd.to_numeric(pd.to_datetime(self.df[col2]))
            else:
                data2 = self.df[col2]

            mask = data1.notna() & data2.notna()
            data1_clean = data1[mask]
            data2_clean = data2[mask]

            if len(data1_clean) < 2:
                return {'Ошибка': 'Недостаточно данных для расчета'}

            # Расчет корреляции Пирсона
            correlation = data1_clean.corr(data2_clean)

            # Интерпретация
            strength = self.interpret_correlation(correlation)
            interpretation = self.get_correlation_interpretation(correlation)

            return {
                'Столбец 1': col1,
                'Столбец 2': col2,
                'Коэффициент корреляции Пирсона': correlation,
                'Сила связи': strength,
                'Интерпретация': interpretation,
                'Количество пар значений': len(data1_clean),
                'p-значение (приблизительно)': self.estimate_p_value(correlation, len(data1_clean))
            }

        except Exception as e:
            return {'Ошибка': str(e)}

    def interpret_correlation(self, r):
        """Интерпретация силы корреляции"""
        r_abs = abs(r)
        if r_abs >= 0.9:
            return "Очень сильная"
        elif r_abs >= 0.7:
            return "Сильная"
        elif r_abs >= 0.5:
            return "Умеренная"
        elif r_abs >= 0.3:
            return "Слабая"
        else:
            return "Очень слабая или отсутствует"

    def get_correlation_interpretation(self, r):
        """Полное описание корреляции"""
        r_abs = abs(r)
        if r_abs >= 0.9:
            return "Практически линейная зависимость"
        elif r_abs >= 0.7:
            return "Сильная статистическая зависимость"
        elif r_abs >= 0.5:
            return "Заметная зависимость"
        elif r_abs >= 0.3:
            return "Слабая зависимость"
        elif r_abs >= 0.1:
            return "Очень слабая зависимость"
        else:
            return "Нет статистически значимой связи"

    def estimate_p_value(self, r, n):
        """Оценка p-значения (упрощенная)"""
        if n <= 2:
            return "Недостаточно данных"

        # Упрощенная оценка значимости
        t_stat = abs(r) * np.sqrt((n - 2) / (1 - r ** 2)) if r != 1 else float('inf')

        # Очень упрощенная оценка p-значения
        if t_stat > 3.5:
            return "< 0.001 (высоко значимо)"
        elif t_stat > 2.6:
            return "< 0.01 (значимо)"
        elif t_stat > 1.96:
            return "< 0.05 (умеренно значимо)"
        else:
            return "> 0.05 (не значимо)"

    def format_correlation_text(self):
        """Форматирование текста для корреляционного анализа"""
        if 'Ошибка' in self.corr_result:
            return f"Ошибка: {self.corr_result['Ошибка']}"

        text = "═" * 80 + "\n"
        text += "КОРРЕЛЯЦИОННЫЙ АНАЛИЗ\n"
        text += f"Между: {self.corr_result['Столбец 1']} и {self.corr_result['Столбец 2']}\n"
        text += "═" * 80 + "\n\n"

        text += "🔗 РЕЗУЛЬТАТЫ КОРРЕЛЯЦИОННОГО АНАЛИЗА:\n"
        text += "─" * 40 + "\n"

        # Основные результаты
        text += f"{'Коэффициент корреляции Пирсона (r)':<45}: {self.corr_result['Коэффициент корреляции Пирсона']:>15.4f}\n"

        # Знак корреляции
        r = self.corr_result['Коэффициент корреляции Пирсона']
        if r > 0:
            direction = "Прямая (положительная)"
        elif r < 0:
            direction = "Обратная (отрицательная)"
        else:
            direction = "Отсутствует"
        text += f"{'Направление связи':<45}: {direction:>15}\n"

        text += f"{'Сила связи':<45}: {self.corr_result['Сила связи']:>15}\n"
        text += f"{'Интерпретация':<45}: {self.corr_result['Интерпретация']:>15}\n"

        text += "\n📊 СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:\n"
        text += "─" * 40 + "\n"

        text += f"{'Количество пар значений (n)':<45}: {self.corr_result['Количество пар значений']:>15}\n"
        text += f"{'p-значение (приблизительно)':<45}: {self.corr_result['p-значение (приблизительно)']:>15}\n"

        # Рекомендации
        text += "\n💡 РЕКОМЕНДАЦИИ:\n"
        text += "─" * 40 + "\n"

        r_abs = abs(r)
        if r_abs >= 0.7:
            text += "• Сильная корреляция позволяет делать прогнозы\n"
            text += "• Зависимость может быть использована для моделирования\n"
        elif r_abs >= 0.5:
            text += "• Умеренная корреляция указывает на заметную связь\n"
            text += "• Может быть полезна для анализа тенденций\n"
        elif r_abs >= 0.3:
            text += "• Слабая корреляция требует осторожности в выводах\n"
            text += "• Рекомендуется дополнительный анализ\n"
        else:
            text += "• Корреляция очень слабая или отсутствует\n"
            text += "• Выводы о связи делать не рекомендуется\n"

        # Математическая интерпретация
        text += "\n📐 МАТЕМАТИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:\n"
        text += "─" * 40 + "\n"

        if r_abs >= 0.9:
            text += f"• r² = {r ** 2:.3f} - {r ** 2 * 100:.1f}% дисперсии объясняется связью\n"
        elif r_abs >= 0.7:
            text += f"• r² = {r ** 2:.3f} - {r ** 2 * 100:.1f}% дисперсии объясняется связью\n"
        elif r_abs >= 0.5:
            text += f"• r² = {r ** 2:.3f} - {r ** 2 * 100:.1f}% дисперсии объясняется связью\n"
        elif r_abs >= 0.3:
            text += f"• r² = {r ** 2:.3f} - {r ** 2 * 100:.1f}% дисперсии объясняется связью\n"
        else:
            text += f"• r² = {r ** 2:.3f} - менее 10% дисперсии объясняется связью\n"

        text += "\n" + "═" * 80 + "\n"
        text += f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += "═" * 80 + "\n"

        return text

    # =========================================================================
    # МЕТОДЫ ДЛЯ ГРАФИКОВ (ПОЛНАЯ РЕАЛИЗАЦИЯ)
    # =========================================================================

    def validate_numeric_column(self, column_name, action_name):
        """Проверка валидности числового столбца"""
        if column_name not in self.df.columns:
            QMessageBox.warning(self, "Ошибка", f"Столбец '{column_name}' не найден")
            return False

        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            QMessageBox.warning(self, "Ошибка",
                                f"Для {action_name} выберите числовой столбец")
            return False

        if self.df[column_name].dropna().empty:
            QMessageBox.warning(self, "Ошибка", f"Столбец '{column_name}' не содержит данных")
            return False

        return True

    def validate_two_columns(self, col1, col2, action_name):
        """Проверка валидности двух столбцов"""
        if col1 not in self.df.columns or col2 not in self.df.columns:
            QMessageBox.warning(self, "Ошибка", "Один из столбцов не найден")
            return False

        if col1 == col2:
            QMessageBox.warning(self, "Ошибка", "Выберите разные столбцы")
            return False

        return True

    # 1. График плотности распределения
    def open_density_plot(self):
        """Открытие окна с графиком плотности распределения"""
        column = self.combo_single.currentText()

        if not self.validate_numeric_column(column, "построения плотности распределения"):
            return

        window = PlotWindow(f"Плотность распределения: {column}", self)
        window.plot_data(self.plot_density, column)
        window.show()
        self.plot_windows.append(window)

    def plot_density(self, ax, column):
        """Функция построения графика плотности распределения"""
        data = self.df[column].dropna()

        plt.style.use('seaborn-v0_8-whitegrid')

        # Гистограмма
        n_bins = min(50, len(data) // 5)
        n_bins = max(10, n_bins)

        ax.hist(data, bins=n_bins, density=True, alpha=0.6,
                color='skyblue', edgecolor='black', linewidth=0.5)

        # KDE plot
        try:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 1000)
            ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
        except:
            pass

        # Вертикальные линии статистик
        mean_val = data.mean()
        median_val = data.median()

        ax.axvline(mean_val, color='red', linestyle='-', linewidth=2,
                   label=f'Среднее: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                   label=f'Медиана: {median_val:.2f}')

        # Мода
        mode_vals = data.mode()
        if not mode_vals.empty:
            for mode_val in mode_vals:
                ax.axvline(mode_val, color='orange', linestyle=':', linewidth=2,
                           label=f'Мода: {mode_val:.2f}')

        ax.set_title(f'Плотность распределения: {column}', fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Плотность', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    # 2. Гистограмма
    def open_histogram_plot(self):
        """Открытие окна с гистограммой"""
        column = self.combo_single.currentText()

        if not self.validate_numeric_column(column, "построения гистограммы"):
            return

        window = PlotWindow(f"Гистограмма: {column}", self)
        window.plot_data(self.plot_histogram, column)
        window.show()
        self.plot_windows.append(window)

    def plot_histogram(self, ax, column):
        """Функция построения гистограммы"""
        data = self.df[column].dropna()

        # Автоматический расчет оптимального количества бинов
        n_bins = self.calculate_optimal_bins(data)

        # Построение гистограммы
        n, bins, patches = ax.hist(data, bins=n_bins, alpha=0.7,
                                   color='lightgreen', edgecolor='darkgreen',
                                   linewidth=1)

        # Добавление кривой нормального распределения
        try:
            mu, sigma = stats.norm.fit(data)
            y = stats.norm.pdf(bins, mu, sigma) * len(data) * (bins[1] - bins[0])
            ax.plot(bins, y, 'r--', linewidth=2, label='Нормальное распределение')
        except:
            pass

        # Добавление среднего значения
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='-', linewidth=2,
                   label=f'Среднее: {mean_val:.2f}')

        ax.set_title(f'Гистограмма: {column}', fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Частота', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        # Добавляем аннотации
        ax.text(0.95, 0.95, f'n = {len(data)}\nбинов = {n_bins}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def calculate_optimal_bins(self, data):
        """Вычисление оптимального количества бинов"""
        n = len(data)
        if n <= 30:
            return min(10, n)
        elif n <= 100:
            return int(np.sqrt(n))
        else:
            return int(1 + 3.322 * np.log10(n))

    # 3. Box Plot (IQR)
    def open_boxplot_iqr(self):
        """Открытие окна с Box Plot (IQR)"""
        column = self.combo_single.currentText()

        if not self.validate_numeric_column(column, "построения Box Plot (IQR)"):
            return

        window = PlotWindow(f"Box Plot (IQR): {column}", self)
        window.plot_data(self.plot_boxplot_iqr, column)
        window.show()
        self.plot_windows.append(window)

    def plot_boxplot_iqr(self, ax, column):
        """Функция построения Box Plot (IQR)"""
        data = self.df[column].dropna()

        # Box plot
        bp = ax.boxplot(data, vert=True, patch_artist=True,
                        widths=0.7, showmeans=True, meanline=True,
                        meanprops=dict(color='red', linewidth=2, linestyle='--'),
                        medianprops=dict(color='darkblue', linewidth=2),
                        boxprops=dict(facecolor='lightblue', alpha=0.7))

        # Расчет статистик
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        median = data.median()

        # Добавление аннотаций
        ax.text(1.25, median, f'Медиана: {median:.2f}',
                va='center', fontsize=10)
        ax.text(1.25, data.mean(), f'Среднее: {data.mean():.2f}',
                va='center', fontsize=10)

        # Показ выбросов
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        if len(outliers) > 0:
            ax.plot(np.ones(len(outliers)), outliers, 'ro', alpha=0.6,
                    markersize=8, label=f'Выбросы ({len(outliers)})')

        ax.set_title(f'Box Plot (IQR): {column}', fontsize=14, fontweight='bold')
        ax.set_ylabel(column, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks([1])
        ax.set_xticklabels([''])
        if len(outliers) > 0:
            ax.legend(loc='upper right', fontsize=10)

    # 4. Box Plot (Mean ± Std)
    def open_boxplot_std(self):
        """Открытие окна с Box Plot (Mean ± Std)"""
        column = self.combo_single.currentText()

        if not self.validate_numeric_column(column, "построения Box Plot (Mean ± Std)"):
            return

        window = PlotWindow(f"Box Plot (Mean ± Std): {column}", self)
        window.plot_data(self.plot_boxplot_std, column)
        window.show()
        self.plot_windows.append(window)

    def plot_boxplot_std(self, ax, column):
        """Функция построения Box Plot (Mean ± Std)"""
        data = self.df[column].dropna()
        mean = data.mean()
        std = data.std()

        # Создаем кастомный boxplot
        rect = Rectangle((0.6, mean - std), 0.8, 2 * std,
                         fill=True, alpha=0.3, color='orange',
                         label=f'Mean ± Std')
        ax.add_patch(rect)

        # Линия среднего
        ax.axhline(mean, color='red', linewidth=3, label=f'Mean: {mean:.2f}')

        # Границы ±std
        ax.axhline(mean - std, color='orange', linestyle='--', linewidth=2)
        ax.axhline(mean + std, color='orange', linestyle='--', linewidth=2)

        # Минимум и максимум
        ax.axhline(data.min(), color='blue', linestyle=':', linewidth=1.5,
                   label=f'Min: {data.min():.2f}')
        ax.axhline(data.max(), color='blue', linestyle=':', linewidth=1.5,
                   label=f'Max: {data.max():.2f}')

        # Точки данных
        y_jitter = np.random.normal(0, 0.02, len(data))
        ax.scatter(np.ones(len(data)) + y_jitter, data, alpha=0.4,
                   color='purple', s=20, label=f'Данные (n={len(data)})')

        ax.set_title(f'Box Plot (Mean ± Std): {column}', fontsize=14, fontweight='bold')
        ax.set_ylabel(column, fontsize=12)
        ax.set_xlim(0.4, 1.6)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=10)

    # 5. Scatter Plot
    def open_scatter_plot(self):
        """Открытие окна с Scatter Plot"""
        col1 = self.combo_corr1.currentText()
        col2 = self.combo_corr2.currentText()

        if not self.validate_two_columns(col1, col2, "построения Scatter Plot"):
            return

        window = PlotWindow(f"Scatter Plot: {col1} vs {col2}", self)
        window.plot_data(self.plot_scatter, col1, col2)
        window.show()
        self.plot_windows.append(window)

    def plot_scatter(self, ax, col1, col2):
        """Функция построения Scatter Plot"""
        # Подготовка данных
        x_data = self.df[col1]
        y_data = self.df[col2]

        # Преобразование времени в числовой формат
        if col1 == 'Время':
            x_data = pd.to_datetime(x_data)
            x_numeric = (x_data - x_data.min()).dt.total_seconds()
        else:
            x_numeric = pd.to_numeric(x_data, errors='coerce')

        if col2 == 'Время':
            y_data = pd.to_datetime(y_data)
            y_numeric = (y_data - y_data.min()).dt.total_seconds()
        else:
            y_numeric = pd.to_numeric(y_data, errors='coerce')

        # Удаление NaN
        mask = x_numeric.notna() & y_numeric.notna()
        x_clean = x_numeric[mask]
        y_clean = y_numeric[mask]

        if len(x_clean) < 2:
            raise ValueError("Недостаточно данных для корреляции")

        # Расчет корреляции
        correlation = x_clean.corr(y_clean)

        # Scatter plot
        scatter = ax.scatter(x_clean, y_clean, alpha=0.6, c=y_clean,
                             cmap='viridis', s=50, edgecolors='black', linewidth=0.5)

        # Линия регрессии
        if len(x_clean) > 1:
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
            ax.plot(x_line, p(x_line), "r-", linewidth=2, alpha=0.8,
                    label=f'Линия регрессии\nR = {correlation:.3f}')

        # Настройки графика
        ax.set_title(f'Scatter Plot: {col1} vs {col2}\nКорреляция Пирсона: {correlation:.3f}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(col1, fontsize=12)
        ax.set_ylabel(col2, fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Добавляем аннотацию
        strength = self.interpret_correlation(correlation)
        ax.text(0.02, 0.98, f'Сила связи: {strength}\nn = {len(x_clean)}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 6-8: Графики прогнозирования

    def open_linear_regression_plot(self):
        """Открытие окна с прогнозом линейной регрессии"""
        self.open_forecast_plot("Линейная регрессия", LinearRegression(), lag=7)

    def open_random_forest_plot(self):
        """Открытие окна с прогнозом Random Forest"""
        self.open_forecast_plot("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42), lag=7)

    def open_decision_tree_plot(self):
        """Открытие окна с прогнозом Decision Tree"""
        self.open_forecast_plot("Decision Tree", DecisionTreeRegressor(random_state=42), lag=7)

    def open_forecast_plot(self, model_name, model, lag=7):
        """Общая функция для открытия окон с прогнозами"""
        # Используем выбранный столбец для статистики
        column = self.combo_single.currentText()

        if not self.validate_numeric_column(column, "прогнозирования"):
            return

        try:
            data = self.df[column].dropna().values

            if len(data) < lag * 2:
                QMessageBox.warning(self, "Ошибка",
                                    f"Недостаточно данных для прогноза. Нужно минимум {lag * 2} значений")
                return

            X, y = [], []
            for i in range(len(data) - lag):
                X.append(data[i:i + lag])
                y.append(data[i + lag])

            X = np.array(X)
            y = np.array(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            window = PlotWindow(f"Прогноз ({model_name}): {column}", self)
            window.plot_data(self.plot_forecast, y_test, y_pred, model_name, column, mae, rmse, r2, lag)
            window.show()
            self.plot_windows.append(window)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка прогнозирования", str(e))

    def plot_forecast(self, ax, y_test, y_pred, model_name, column, mae, rmse, r2, lag):
        """Функция построения графика прогноза"""
        indices = range(len(y_test))

        ax.plot(indices, y_test, 'b-', label='Реальные значения', linewidth=2, marker='o', markersize=4)
        ax.plot(indices[:len(y_pred)], y_pred, 'r--', label='Предсказанные значения',
                linewidth=2, marker='s', markersize=4)

        ax.set_title(f'Прогноз: {model_name}\n{column} (лаг={lag})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Индекс тестовых данных', fontsize=12)
        ax.set_ylabel(column, fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        metrics_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nЛаг: {lag}'
        ax.text(0.02, 0.98, metrics_text,
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    def log_message(self, message):
        """Логирование сообщений в консоль"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.append(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        for window in self.plot_windows + self.text_windows:
            window.close()

        reply = QMessageBox.question(
            self, 'Подтверждение',
            'Вы уверены, что хотите закрыть приложение?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)

    app.setStyle('Fusion')

    window = StatisticalAnalyzer()
    window.show()

    window.log_message("═" * 80)
    window.log_message("ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ:")
    window.log_message("═" * 80)
    window.log_message("1. Нажмите 'Загрузить Excel файл' для выбора файла")
    window.log_message("2. Файл должен содержать 3 столбца: Время, Активность, Объем воды (л)")
    window.log_message("3. Выберите столбцы для анализа в выпадающих списках")
    window.log_message("4. Нажмите 'Рассчитать данные' для выполнения расчетов")
    window.log_message("5. Используйте кнопки анализа для просмотра результатов:")
    window.log_message("   • 📊 Статистический анализ - основные статистики")
    window.log_message("   • 🔗 Корреляционный анализ - связь между столбцами")
    window.log_message("6. Нажмите любую кнопку графика для открытия в отдельном окне")
    window.log_message("7. Можно открывать несколько окон одновременно")
    window.log_message("8. Всего доступно 10 окон: 2 текстовых + 8 графических")
    window.log_message("═" * 80)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()