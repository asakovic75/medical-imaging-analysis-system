import gradio as gr
import cv2
import torch
import numpy as np
import os
import time
import segmentation_models_pytorch as smp
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN:
    client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)
    AI_AVAILABLE = True
else:
    client = None
    AI_AVAILABLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=3).to(device)
model_path = "unified_brain_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

def create_colored_overlay(img_res, masks):
    overlay = img_res.copy()
    h, w = overlay.shape[:2]
    for class_id, mask in masks.items():
        if np.sum(mask) > 50:
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
            color = (255, 0, 0) if class_id == 1 else (0, 0, 255)
            colored_mask[mask == 1] = color
            overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay

def predict(file_path):
    if not file_path or not os.path.exists(file_path):
        return None, None, "<div style='text-align:center'>❌ Файл не найден</div>"
    input_img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    start_time = time.time()
    img_resized = cv2.resize(input_img, (256, 256))
    x = (img_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs, axis=0)
        masks = {1: (pred_class == 1).astype(np.uint8), 2: (pred_class == 2).astype(np.uint8)}
    speed_ms = round((time.time() - start_time) * 1000, 1)
    s_area, t_area = np.sum(masks[1])/(256*256)*100, np.sum(masks[2])/(256*256)*100
    s_prob = np.mean(probs[1][masks[1] > 0])*100 if s_area > 0.1 else 0
    t_prob = np.mean(probs[2][masks[2] > 0])*100 if t_area > 0.1 else 0
    has_s, has_t = s_area > 0.3, t_area > 0.3
    active_mask = masks[1] if has_s else (masks[2] if has_t else None)
    side = "Левое полушарие" if active_mask is not None and np.sum(active_mask[:, :128]) > np.sum(active_mask[:, 128:]) else ("Правое полушарие" if active_mask is not None else "Не выявлено")
    if has_s and has_t: verdict, color, area_str, confidence = "СМЕШАННАЯ ПАТОЛОГИЯ", "#CE93D8", f"И:{s_area:.2f}% О:{t_area:.2f}%", f"{(s_prob+t_prob)/2:.0f}%"
    elif has_s: verdict, color, area_str, confidence = "ИНСУЛЬТ", "#FF0000", f"{s_area:.2f}%", f"{s_prob:.1f}%"
    elif has_t: verdict, color, area_str, confidence = "ОПУХОЛЬ", "#0000FF", f"{t_area:.2f}%", f"{t_prob:.1f}%"
    else: verdict, color, area_str, confidence = "НОРМА", "#00FF00", "0.00%", "100.0%"
    res_img = cv2.resize(create_colored_overlay(img_resized, masks), (500, 500))
    orig_img = cv2.resize(input_img, (500, 500))
    html = f'''
    <div style="background: #1a1a2e; border-radius: 16px; padding: 25px; border: 1px solid #7C4DFF; text-align: center; margin-top: 20px;">
        <div style="font-size: 2.2em; font-weight: bold; color: {color}; margin-bottom: 15px;">{verdict}</div>
        <div style="color: #E0E0E0; font-size: 1.1em; line-height: 1.8;">
            <div>📍 Полушарие: <b>{side}</b></div>
            <div>📊 Площадь поражения: <b>{area_str}</b></div>
            <div>🎯 Достоверность: <b>{confidence}</b></div>
            <div>⚡ Скорость анализа: <b>{speed_ms} мс</b></div>
        </div>
    </div>
    '''
    return res_img, orig_img, html

def ask_ai(question, chat_history, diag_ctx):
    if not AI_AVAILABLE:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": "❌ ИИ недоступен."})
        return chat_history
    system_prompt = """Ты — эксперт системы Stroke Diagnosis. Консультируй по симптомам, реабилитации и больницам (Гродно: Университетская БЛК 52, БСМП; Минск: РНПЦ Неврологии). 
    Если вопрос - ерунда или не по медицине, отвечай ровно так: 'Извините, вопрос некорректен. Я специализируюсь на диагностике КТ головного мозга и могу помочь с консультацией по больницам, симптомам или расшифровке. Возможно, вы хотите спросить о чем-то в этой области?'
    Данные анализа: """ + diag_ctx
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history: messages.append(msg)
    messages.append({"role": "user", "content": question})
    try:
        res = client.chat_completion(messages=messages, max_tokens=500)
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": res.choices[0].message.content})
    except:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": "❌ Ошибка ИИ."})
    return chat_history

css = """
.gradio-container {background: #0d0b1a !important; max-width: 98% !important;} 
.header-text {text-align: center; margin-bottom: 20px;}
.header-text h1 {color: #FFFFFF !important; font-size: 2.5em !important; margin-bottom: 0px !important;}
.header-text p {color: #FFFFFF !important; font-size: 1.2em !important; opacity: 1;}
.gr-chatbot {background: #1E1B2E !important; border: 1px solid #7C4DFF !important;} 
footer {display: none !important;}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_classes="header-text"):
        gr.Markdown("# 🧠 Диагностика КТ головного мозга")
        gr.Markdown("Интеллектуальная система автоматического поиска и сегментации инсультов и новообразований")
    file_input = gr.File(label="📂 Загрузите снимок", file_types=[".jpg", ".png", ".jpeg"])
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                out_orig, out_seg = gr.Image(label="ИСХОДНЫЙ СНИМОК", height=500), gr.Image(label="РЕЗУЛЬТАТ СЕГМЕНТАЦИИ (ИИ)", height=500)
            out_html = gr.HTML(value="<div style='text-align:center; color:#555; padding:40px; border:1px dashed #444; border-radius:16px; margin-top:20px;'>Ожидание загрузки снимка...</div>")
            with gr.Row():
                btn_run, btn_clear = gr.Button("🔍 ЗАПУСТИТЬ АНАЛИЗ", variant="primary", size="lg"), gr.Button("🗑 ОЧИСТИТЬ ЭКРАН", variant="secondary", size="lg")
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 ИИ КОНСУЛЬТАНТ")
            chat = gr.Chatbot(height=570)
            msg = gr.Textbox(placeholder="Ваш вопрос...", lines=2)
            btn_ai = gr.Button("💬 ОТПРАВИТЬ", variant="secondary")
    p_st, c_st = gr.State(""), gr.State("")
    file_input.change(lambda f: (cv2.cvtColor(cv2.imread(f.name), cv2.COLOR_BGR2RGB), f.name) if f else (None, ""), [file_input], [out_orig, p_st])
    btn_run.click(predict, [p_st], [out_seg, out_orig, out_html]).then(lambda s, o, h: h, [out_seg, out_orig, out_html], [c_st])
    btn_clear.click(lambda: [None, None, None, "", "", []], None, [file_input, out_orig, out_seg, out_html, msg, chat])
    btn_ai.click(ask_ai, [msg, chat, c_st], [chat]).then(lambda: "", None, [msg])
    msg.submit(ask_ai, [msg, chat, c_st], [chat]).then(lambda: "", None, [msg])

demo.launch(theme=gr.themes.Soft(), css=css)
