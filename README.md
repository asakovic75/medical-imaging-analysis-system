# 🧠 Диагностика инсульта по КТ

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/🤗-Llama%203-黄色)](https://huggingface.co/)

## 📋 О проекте

Автоматическая диагностика КТ-снимков головного мозга с визуализацией инсульта и ИИ-консультантом на базе Llama-3.

### ✨ Возможности системы

Система автоматически распознаёт на КТ-снимках инсульт (красная подсветка и красный текст) или норму (зелёный текст без выделений). Встроенный ИИ-консультант на базе Llama-3 отвечает на вопросы о диагнозе, симптомах, реабилитации и клиниках Беларуси. Анализ занимает около 1 секунды на один снимок — быстро, точно, наглядно.

## 🛠 Технологии

| Компонент | Технология |
|-----------|-------------|
| Нейросеть | U-Net + EfficientNet-B4 |
| Фреймворк | PyTorch |
| Интерфейс | Gradio |
| ИИ-чат | Hugging Face Inference API (Meta-Llama-3-8B-Instruct) |

## 📁 Структура проекта
* app.py — главный файл 
* stroke_model.pth — веса обученной нейросети
* requirements.txt — зависимости
* README.md — документация

## 🚀 Быстрый старт

### Онлайн (без установки)

👉 **[Открыть демо на Hugging Face Spaces](https://asakovic75-multict.hf.space)**

### Локально

```bash
# 1. Клонировать репозиторий
git clone https://github.com/asakovic75/stroke-diagnosis-pro.git
cd stroke-diagnosis-pro

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Запустить приложение
python app.py
