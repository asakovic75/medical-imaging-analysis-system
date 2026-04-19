# 🧠 Интеллектуальная система анализа и хранения медицинских изображений

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/asakovic75/stroke-diagnosis)

## 📋 О проекте
Интеллектуальная система для автоматической диагностики КТ-снимков головного мозга. Система находит патологии, сохраняет результаты в базу данных и выдает готовый медицинский отчет.

### ✨ Основные возможности:
*   **ИИ-диагностика.** Распознает **Инсульт**, **Опухоль** или **Норму**.
*   **Сегментация.** Подсвечивает зону поражения на снимке.
*   **База данных.** Автоматически ведет таблицу обследований (ID, площадь, вердикт) с экспортом в **CSV**.
*   **PDF-отчеты.** Генерирует протокол обследования для врача.
*   **ИИ-консультантю** Чат-бот (Llama-3) для ответов на вопросы и навигации по клиникам Беларуси.

## 🛠 Технологии
*   **Компьютерное зрение.** U-Net (EfficientNet-B4) на PyTorch.
*   **Интерфейс.** Gradio.
*   **Анализ данных.** Pandas (хранение истории).
*   **Отчеты.** FPDF2.

## 📁 Структура файлов
*   `app.py` / `logic.py` — исходный код системы.
*   `MULTI.ipynb` / `STROKE_NOTEBOOK.ipynb` — **обучение моделей**.
*   `stroke_history.csv` — локальная база данных.
*   `unified_brain_model.pth` — веса нейросети.

## 🚀 Как запустить

### Онлайн (самый быстрый способ):
👉 **[Открыть Live Demo в браузере](https://huggingface.co/spaces/asakovic75/stroke-diagnosis)**

### Локально:
```bash
git clone https://github.com/asakovic75/medical-imaging-analysis-system.git
cd medical-imaging-analysis-system
pip install -r requirements.txt
python app.py
