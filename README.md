# 🧠 Диагностика КТ головного мозга

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/🤗-Llama%203-黄色)](https://huggingface.co/)

## 📋 О проекте

Автоматическая диагностика КТ-снимков головного мозга с визуализацией патологий и ИИ-консультантом на базе Llama-3.

### ✨ Возможности

🔴 ИНСУЛЬТ — Красная подсветка на снимке, красный текст диагноза
🔵 ОПУХОЛЬ — Синяя подсветка на снимке, синий текст диагноза
🟢 НОРМА — Зеленый текст, без выделений
🟣 СМЕШАННАЯ ПАТОЛОГИЯ — Фиолетовый текст диагноза
🤖 ИИ-КОНСУЛЬТАНТ — Llama-3 отвечает на вопросы, знает клиники Беларуси
⚡ БЫСТРЫЙ АНАЛИЗ — ~1 секунда на снимок

## 🛠 Технологии

| Компонент | Технология |
|-----------|-------------|
| Нейросеть | U-Net + EfficientNet-B4 |
| Фреймворк | PyTorch |
| Интерфейс | Gradio |
| ИИ-чат | Hugging Face Inference API (Meta-Llama-3-8B-Instruct) |

## 📁 Структура
medical-imaging-analysis-system/
* app.py # Главный файл (вся логика здесь)
* unified_brain_model.pth # Веса обученной нейросети
* requirements.txt # Зависимости
* README.md # Документация


## 🚀 Быстрый старт

### Онлайн (без установки)

👉 **[Открыть демо на Hugging Face Spaces](https://asakovic75-multi.hf.space/)**

### Локально

```bash
# 1. Клонировать репозиторий
git clone https://github.com/asakovic75/medical-imaging-analysis-system.git
cd medical-imaging-analysis-system

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Запустить приложение
python app.py
