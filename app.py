import torch
import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from datetime import datetime

# ==================== تحميل النموذج ====================
BASE_MODEL = "sherif1313/3arabLM-4B-Fiqh-v1"

print("⏳ تحميل النموذج...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("✅ تم تحميل النموذج بنجاح")

# ==================== ملف التعليقات ====================
FEEDBACK_FILE = "feedback.jsonl"

def load_comments(as_html=False):
    if not os.path.exists(FEEDBACK_FILE):
        return "📭 لا توجد تعليقات حتى الآن. كن أول من يشارك! 💬"
    
    comments = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if as_html:
                    comment_html = f"""
                    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 8px 0; background: #f9f9f9;">
                        <div style="display: flex; justify-content: space-between; color: #555;">
                            <span>⭐ {data.get('rating', '?')}/5</span>
                            <span style="font-size: 0.9em;">{data.get('timestamp', '')}</span>
                        </div>
                        <div style="margin: 6px 0; font-weight: bold;">📝 {data.get('comment', '')}</div>
                        <div style="font-size: 0.95em; color: #333; background: #f0f0f0; padding: 4px 8px; border-radius: 4px;">
                            <span style="color: #888;">السؤال:</span> {data.get('user_prompt', '')[:150]}...
                        </div>
                        <div style="font-size: 0.95em; color: #333; background: #f0f0f0; padding: 4px 8px; border-radius: 4px; margin-top: 4px;">
                            <span style="color: #888;">الإجابة:</span> {data.get('model_answer', '')[:150]}...
                        </div>
                    </div>
                    """
                    comments.append(comment_html)
                else:
                    comment_text = (
                        f"⭐ {data.get('rating', '?')}/5 | {data.get('timestamp', '')}\n"
                        f"📝 {data.get('comment', '')[:200]}\n"
                        f"🔹 السؤال: {data.get('user_prompt', '')[:100]}...\n"
                        f"🔸 الإجابة: {data.get('model_answer', '')[:150]}...\n"
                        f"{'─' * 50}\n"
                    )
                    comments.append(comment_text)
            except:
                continue
    
    if not comments:
        return "📭 لا توجد تعليقات حتى الآن. كن أول من يشارك! 💬"
    
    if as_html:
        return "".join(comments[-10:][::-1])
    else:
        return "\n".join(comments[-20:][::-1])

def save_comment_and_refresh(comment, rating, user_prompt, model_answer):
    if not comment or comment.strip() == "":
        return "⚠️ الرجاء كتابة تعليق قبل الإرسال.", load_comments(as_html=True)
    
    feedback_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rating": rating,
        "comment": comment.strip(),
        "user_prompt": user_prompt[:300],
        "model_answer": model_answer[:500]
    }
    
    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
        status_msg = "✅ شكراً لك! تم إضافة تعليقك بنجاح."
    except Exception as e:
        status_msg = f"❌ خطأ في الحفظ: {e}"
    
    updated_comments = load_comments(as_html=True)
    return status_msg, updated_comments

# ==================== دالة التوليد ====================
@spaces.GPU(duration=120)
def generate_text(prompt, book, author, category, max_new_tokens):
    full_prompt = f"book {book}\nauthor {author}\ncategory {category}\n{prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.15,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.08,
            no_repeat_ngram_size=5,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return generated

# ==================== خيارات التصنيف ====================
CATEGORIES = ["التفاسير", "فقه عام", "نحو", "بدون"]

# ==================== تصميم واجهة Gradio ====================
with gr.Blocks(title="محرك النصوص التراثية نسخة تجريبيه", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📖 3arabLM-4B | محرك النصوص التراثية")
    gr.Markdown("💬 **التعليقات مرئية للجميع** - ساعدنا في تحسين النموذج بمشاركة رأيك.")

    with gr.Row():
        with gr.Column(scale=1):
            book_input = gr.Textbox(label="اسم الكتاب", value="التفسير", placeholder="مثال: التفسير")
            author_input = gr.Textbox(label="اسم المؤلف", value="ابن كثير", placeholder="مثال: ابن كثير")
            category_dropdown = gr.Dropdown(choices=CATEGORIES, label="التصنيف", value="التفاسير")
            max_tokens_slider = gr.Slider(
                minimum=50, maximum=512, value=300, step=50,
                label="📏 الطول الأقصى للنص المولد (Tokens)"
            )

        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="النص / السؤال",
                placeholder="مثال: بسم الله الرحمن الرحيم، قال الله تعالى: {الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ} أي...",
                lines=8
            )
            generate_btn = gr.Button("🚀 توليد النص", variant="primary")
            output_text = gr.Textbox(label="الناتج", lines=12)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 💬 أضف تعليقك")
            feedback_rating = gr.Slider(1, 5, value=5, step=1, label="⭐ التقييم")
            feedback_comment = gr.Textbox(
                label="تعليقك",
                placeholder="اكتب ملاحظاتك حول الإجابة، أخطاء لغوية، اقتراحات...",
                lines=3
            )
            feedback_btn = gr.Button("📨 إرسال التعليق", variant="secondary")
            feedback_status = gr.Textbox(label="حالة الإرسال", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 📋 التعليقات السابقة")
            comments_display = gr.HTML(value=load_comments(as_html=True), label="")
            refresh_btn = gr.Button("🔄 تحديث التعليقات", size="sm")

    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, book_input, author_input, category_dropdown, max_tokens_slider],
        outputs=output_text
    )

    feedback_btn.click(
        fn=save_comment_and_refresh,
        inputs=[feedback_comment, feedback_rating, prompt_input, output_text],
        outputs=[feedback_status, comments_display]
    )

    refresh_btn.click(
        fn=lambda: load_comments(as_html=True),
        inputs=[],
        outputs=comments_display
    )

if __name__ == "__main__":
    demo.launch()
