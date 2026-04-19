import gradio as gr
import pandas as pd
import logic

css = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
#header { text-align: center !important; }
.ai-terminal {
    background-color: #1a202c !important; color: #f7fafc !important; padding: 25px !important;
    border-radius: 15px !important; border: 2px solid #4a5568 !important; font-family: 'Courier New', monospace !important; line-height: 1.6 !important;
}
.ai-terminal * { color: #f7fafc !important; }
footer { display: none !important; }
.compact-df { margin-top: -10px !important; }
"""

with gr.Blocks(fill_width=True, css=css) as demo:
    gr.Markdown("<div id='header'><h1>🧠 Диагностика патологий головного мозга по КТ</h1><h3>Интеллектуальная система анализа медицинских изображений</h3></div>")
    
    with gr.Column():
        model_selector = gr.Dropdown(choices=list(logic.model_paths.keys()), value=list(logic.model_paths.keys())[0] if logic.model_paths else None, label="🔧 ВЫБЕРИТЕ НЕЙРОСЕТЕВУЮ МОДЕЛЬ")
        input_f = gr.Image(label="📸 ЗАГРУЗИТЕ СНИМОК КТ", type="filepath")
        with gr.Row():
            btn = gr.Button("🔍 ЗАПУСТИТЬ АНАЛИЗ", variant="primary", size="lg")
            clr = gr.ClearButton(value="🗑 ОЧИСТИТЬ ЭКРАН", size="lg")
        status_out, details_out = gr.HTML(), gr.HTML()
        with gr.Row():
            o_res = gr.Image(label="🎯 Результат сегментации (ИИ)", height=400, width=400)
            o_orig = gr.Image(label="📷 Исходный снимок", height=400, width=400)
        pdf_file = gr.File(label="📄 МЕДИЦИНСКИЙ ОТЧЕТ (PDF)")
        
        gr.HTML("<br>")
        with gr.Column(elem_classes="ai-terminal"):
            gr.Markdown("### 🤖 ИИ-АССИСТЕНТ")
            ai_q1 = gr.Textbox(label="Вопрос по снимку", placeholder="Каковы риски? Куда направить пациента в Гродно?")
            ai_btn1 = gr.Button("💬 ПОЛУЧИТЬ КОНСУЛЬТАЦИЮ ИИ", variant="secondary")
            ai_out1 = gr.Markdown("💻 Ожидание запроса...")
        gr.HTML("<br>")
        
        history_table = gr.Dataframe(value=pd.DataFrame(logic.history_list, columns=logic.COLUMNS), interactive=False, elem_classes="compact-df")
        with gr.Row():
            save_csv_btn = gr.Button("💾 СОХРАНИТЬ (CSV)", variant="primary", size="lg")
            download_csv_btn = gr.DownloadButton("📥 СКАЧАТЬ (CSV)", size="lg")
            
    btn.click(logic.predict_multiclass, [input_f, model_selector], [o_res, o_orig, status_out, details_out, history_table, pdf_file])
    ai_btn1.click(lambda q: logic.ask_ai_assistant(q, "Клинический анализ", logic.current_context), [ai_q1], [ai_out1])
    clr.add([input_f, o_res, o_orig, status_out, details_out, history_table, pdf_file, ai_q1, ai_out1])

    save_csv_btn.click(lambda: gr.Info("История сохранена!"), None, None)
    download_csv_btn.click(lambda: logic.DB_PATH, None, download_csv_btn)

if __name__ == "__main__":
    demo.launch(ssr_mode=False, theme=gr.themes.Soft())
