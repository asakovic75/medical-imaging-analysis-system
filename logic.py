import os, cv2, torch, numpy as np, time, pandas as pd
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import segmentation_models_pytorch as smp
import pydicom
from huggingface_hub import InferenceClient
import pytz

HF_TOKEN = os.environ.get("HF_TOKEN", None)

if HF_TOKEN:
    client = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token=HF_TOKEN
    )
    AI_AVAILABLE = True
else:
    client = None
    AI_AVAILABLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DB_PATH = "stroke_history.csv"
FONT_PATH = "DejaVuSans.ttf"

COLUMNS = ["ID", "Снимок", "Дата", "Время", "Модель", "Вердикт", "Тип патологии", "Локализация", "Площадь (%)", "Достоверность", "Скорость (мс)"]

allowed = ["unified_brain_model.pth"]
model_paths = {f"🧠 {name}": name for name in allowed if os.path.exists(name)}
if not model_paths:
    model_paths["❌ Модели не найдены"] = None

COLORS = {
    "НОРМА": "#2E7D32",
    "ИНСУЛЬТ": "#D32F2F",
    "ОПУХОЛЬ": "#2196F3",
    "СМЕШАННАЯ": "#9C27B0"
}

current_context = ""

def load_database():
    if os.path.exists(DB_PATH):
        try:
            df = pd.read_csv(DB_PATH)
            if len(df.columns) == len(COLUMNS):
                df['ID'] = pd.to_numeric(df['ID'])
                return df.values.tolist()
        except:
            pass
    return []

history_list = load_database()

model = smp.Unet(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=3
).to(device)

def load_selected_model(model_key):
    if not model_key or "❌" in model_key:
        return False
    try:
        model.load_state_dict(torch.load(model_paths[model_key], map_location=device))
        model.eval()
        return True
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return False

def ask_ai_assistant(question, context_type, data_summary):
    if not AI_AVAILABLE or client is None:
        return "⚠️ AI ассистент недоступен. Добавьте HF_TOKEN в Secrets."
    
    if len(question.strip()) < 3:
        return "❌ Ошибка: Некорректный запрос"
    
    system_prompt = """Ты — врач-радиолог и нейрохирург экспертного центра.
СПЕЦИАЛИЗАЦИЯ: КТ диагностика головного мозга, инсульты, опухоли, нормы.

ДИАГНОСТИКА ПО КТ:
- Ишемический инсульт: темная зона (гиподенсивная)
- Геморрагический инсульт: ярко-белая зона (гиперденсивная)
- Опухоль: гипо/гиперденсивная, часто с перифокальным отеком

КЛИНИКИ ГРОДНО:
- Университетская клиника (БЛК 52) - экстренная помощь, КТ 24/7
- БСМП (ул. Советских Пограничников 115) - первичное сосудистое отделение
- Городская больница №2 (Гагарина 5) - реабилитация

КЛИНИКИ БЕЛАРУСИ (МИНСК):
- РНПЦ Неврологии и Нейрохирургии (ул. Филимонова 43) - главный центр
- РНПЦ Онкологии им. Александрова (п. Лесной) - лечение опухолей мозга
- 6-я городская больница (ул. Уборевича 177) - сосудистый центр

На вопросы не по теме отвечай: 'Ошибка: Запрос не относится к медицинской области'"""

    full_prompt = f"{system_prompt}\n\nДАННЫЕ ПАЦИЕНТА: {data_summary}\n\nВОПРОС: {question}"
    
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Ошибка связи с ИИ: {str(e)[:100]}"

def generate_pdf_report(orig_img, seg_img, info):
    pdf = FPDF()
    has_font = os.path.exists(FONT_PATH)
    if has_font:
        try:
            pdf.add_font("DejaVu", "", FONT_PATH)
            pdf.add_font("DejaVu", "B", FONT_PATH)
        except:
            has_font = False

    pdf.add_page()
    
    if has_font:
        pdf.set_font("DejaVu", "B", 18)
        pdf.cell(0, 12, "МЕДИЦИНСКИЙ ОТЧЕТ КТ", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("DejaVu", "", 10)
        pdf.cell(0, 7, f"ID Пациента: {info['p_id']} | Снимок №{info['p_id']} | Файл: {info['filename']}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 7, f"Дата: {info['date']} | Время: {info['time']}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(6)

        pdf.set_font("DejaVu", "B", 14)
        pdf.cell(0, 10, "1. РЕЗУЛЬТАТЫ ОБСЛЕДОВАНИЯ:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("DejaVu", "", 11)
        
        if info['pathology_type'] == "Инсульт":
            pdf.set_text_color(211, 47, 47)
        elif info['pathology_type'] == "Опухоль":
            pdf.set_text_color(33, 150, 243)
        elif info['pathology_type'] == "Норма":
            pdf.set_text_color(46, 125, 50)
        else:
            pdf.set_text_color(156, 39, 176)
        
        pdf.cell(0, 8, f"- Заключение: {info['verdict']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        
        pdf.cell(0, 8, f"- Тип патологии: {info['pathology_type']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 8, f"- Локализация: {info['side']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 8, f"- Площадь поражения: {info['area']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 8, f"- Достоверность: {info['confidence']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 8, f"- Скорость анализа: {info['speed']} мс", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 8, f"- Модель: {info['model']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        cv2.imwrite("_orig_tmp.jpg", cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("_seg_tmp.jpg", cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
        
        pdf.image("_orig_tmp.jpg", x=20, y=130, w=80)
        pdf.image("_seg_tmp.jpg", x=110, y=130, w=80)

        pdf.set_y(-25)
        pdf.set_font("DejaVu", "", 7)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 4, "ВНИМАНИЕ: Данный отчет сформирован системой ИИ. Он носит справочный характер и не является диагнозом.", align="C")
    else:
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "MEDICAL REPORT", align="C")
        pdf.ln(10)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, f"Patient ID: {info['p_id']}\nImage No: {info['p_id']}\nFile: {info['filename']}\nDate: {info['date']}\n\nDiagnosis: {info['verdict']}\nPathology: {info['pathology_type']}\nSide: {info['side']}\nArea: {info['area']}\nConfidence: {info['confidence']}\nSpeed: {info['speed']} ms\nModel: {info['model']}")

    output_name = f"medical_report_{info['p_id']}.pdf"
    pdf.output(output_name)
    return output_name

def create_colored_overlay(img_res, masks, probs):
    overlay = img_res.copy()
    h, w = overlay.shape[:2]
    
    colors = {
        1: (0, 0, 255),
        2: (255, 140, 0)
    }
    
    for class_id, color in colors.items():
        if class_id in masks:
            mask = masks[class_id]
            mask_area = np.sum(mask)
            
            if mask_area > 50:
                colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
                colored_mask[mask == 1] = color
                alpha = 0.5
                overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
    
    return overlay

def resize_to_square(img, size=400):
    h, w = img.shape[:2]
    if h == w and h == size:
        return img
    if h > w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)
    resized = cv2.resize(img, (new_w, new_h))
    square = np.ones((size, size, 3), dtype=np.uint8) * 0
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return square

def predict_multiclass(file_path, model_key):
    global history_list, current_context
    
    if not file_path or not load_selected_model(model_key):
        return [None] * 6
    
    filename = os.path.basename(file_path)
    input_img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    
    start_time = time.time()
    
    img_resized = cv2.resize(input_img, (256, 256))
    x = (img_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs, axis=0)
        
        masks = {
            1: (pred_class == 1).astype(np.uint8),
            2: (pred_class == 2).astype(np.uint8)
        }
    
    speed_ms = round((time.time() - start_time) * 1000, 1)
    
    stroke_area = np.sum(masks[1]) / (256 * 256) * 100
    tumor_area = np.sum(masks[2]) / (256 * 256) * 100
    
    stroke_prob = np.mean(probs[1][masks[1] > 0]) if stroke_area > 0.1 else 0
    tumor_prob = np.mean(probs[2][masks[2] > 0]) if tumor_area > 0.1 else 0
    
    has_stroke = stroke_area > 0.3
    has_tumor = tumor_area > 0.3
    
    if has_stroke and has_tumor:
        verdict = "СМЕШАННАЯ ПАТОЛОГИЯ"
        pathology_type = "Инсульт + Опухоль"
        confidence = f"И:{stroke_prob*100:.0f}% / О:{tumor_prob*100:.0f}%"
        area_str = f"И:{stroke_area:.2f}% / О:{tumor_area:.2f}%"
        active_mask = masks[1] if stroke_area > tumor_area else masks[2]
    elif has_stroke:
        verdict = "ОБНАРУЖЕН ИНСУЛЬТ"
        pathology_type = "Инсульт"
        confidence = f"{stroke_prob*100:.1f}%"
        area_str = f"{stroke_area:.2f}%"
        active_mask = masks[1]
    elif has_tumor:
        verdict = "ОБНАРУЖЕНА ОПУХОЛЬ"
        pathology_type = "Опухоль"
        confidence = f"{tumor_prob*100:.1f}%"
        area_str = f"{tumor_area:.2f}%"
        active_mask = masks[2]
    else:
        verdict = "НОРМА"
        pathology_type = "Норма"
        confidence = "100.0%"
        area_str = "0.00%"
        active_mask = None
    
    if active_mask is not None:
        left_sum = np.sum(active_mask[:, :128])
        right_sum = np.sum(active_mask[:, 128:])
        if left_sum > right_sum:
            side = "Левое полушарие"
        else:
            side = "Правое полушарие"
    else:
        side = "Не выявлено"
    
    seg_overlay = create_colored_overlay(img_resized, masks, probs)
    
    seg_overlay_resized = cv2.resize(seg_overlay, (400, 400))
    input_img_resized = resize_to_square(input_img, 400)
    
    p_id = int(max([row[0] for row in history_list])) + 1 if history_list else 1
    now_gr = datetime.now(pytz.timezone('Europe/Minsk'))
    
    info = {
        'p_id': p_id,
        'filename': filename,
        'model': model_key.replace("🧠 ", ""),
        'verdict': verdict,
        'pathology_type': pathology_type,
        'side': side,
        'area': area_str,
        'confidence': confidence,
        'speed': speed_ms,
        'date': now_gr.strftime("%d.%m.%Y"),
        'time': now_gr.strftime("%H:%M:%S")
    }
    
    current_context = f"Снимок: {filename}. Диагноз: {verdict}. Тип: {pathology_type}. Локализация: {side}. Площадь: {area_str}. Достоверность: {confidence}."
    
    pdf_file = generate_pdf_report(img_resized, seg_overlay, info)
    
    history_list.insert(0, [
        p_id, filename, info['date'], info['time'], info['model'],
        verdict, pathology_type, side, area_str, confidence, f"{speed_ms}"
    ])
    pd.DataFrame(history_list, columns=COLUMNS).to_csv(DB_PATH, index=False)
    
    if pathology_type == "Норма":
        color = COLORS["НОРМА"]
    elif pathology_type == "Инсульт":
        color = COLORS["ИНСУЛЬТ"]
    elif pathology_type == "Опухоль":
        color = COLORS["ОПУХОЛЬ"]
    else:
        color = COLORS["СМЕШАННАЯ"]
    
    status_html = f'<div style="text-align: center; font-size: 2em; font-weight: bold; color: {color}; padding: 10px;">{verdict}</div>'
    
    details_html = f'''
    <div style="text-align: center; padding: 15px; line-height: 2; background-color: transparent; max-width: 450px; margin: 0 auto;">
        <div style="font-size: 1.1em;">📋 <b>Тип патологии:</b> {pathology_type}</div>
        <div style="font-size: 1.1em;">🎯 <b>Достоверность:</b> {confidence}</div>
        <div style="font-size: 1.1em;">📊 <b>Площадь поражения:</b> {area_str}</div>
        <div style="font-size: 1.1em;">📍 <b>Локализация:</b> {side}</div>
        <div style="font-size: 1.1em;">⚙️ <b>Модель:</b> {info['model']}</div>
        <div style="color: #666; font-size: 0.9em; margin-top: 10px;">🆔 ID: {p_id} | Снимок №{p_id} | ⚡ {speed_ms} мс</div>
    </div>
    '''
    
    df_history = pd.DataFrame(history_list, columns=COLUMNS)
    
    return seg_overlay_resized, input_img_resized, status_html, details_html, df_history, pdf_file