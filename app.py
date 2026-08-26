import torch
import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from datetime import datetime

# ==================== تحميل النموذج ====================
BASE_MODEL = "sherif1313/3arabLM-4B-islamic-v2"  # غيّره إذا كان الاسم مختلفاً

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

# ==================== التصنيفات الجديدة (مطابقة لكود التدريب) ====================
CATEGORIES_MAIN = [
    "التفاسير",
    "فقه عام",
    "متون الحديث",
    "العقيدة",
    "النحو والصرف",
    "الفتاوى"
]

# ==================== بيانات الكتب (25 كتاباً) ====================
# تم تعديل التصنيفات لتطابق القائمة أعلاه
BOOKS_DATA = {
    # ✍️ النحو والصرف (5 كتب)
    "النحو الوافي": ("عباس حسن", "النحو والصرف"),
    "تمهيد القواعد بشرح تسهيل الفوائد": ("ابن مالك", "النحو والصرف"),
    "شرح ألفية ابن مالك للحازمي": ("الحازمي", "النحو والصرف"),
    "شرح ألفية ابن مالك للشاطبي = المقاصد الشافية": ("الشاطبي", "النحو والصرف"),
    "شرح المفصل لابن يعيش": ("ابن يعيش", "النحو والصرف"),

    # ⚖️ فقه عام (3 كتب)
    "الموسوعة الفقهية الكويتية": ("وزارة الأوقاف الكويتية", "فقه عام"),
    "موسوعة الإجماع في الفقه الإسلامي": ("مجموعة مؤلفين", "فقه عام"),
    "موسوعة فقه العبادات": ("مجموعة مؤلفين", "فقه عام"),

    # ⚖️ الفتاوى (2 كتب أساسية + 1 إضافي)
    "فتاوى الشبكة الإسلامية": ("الشبكة الإسلامية", "الفتاوى"),
    "مجموع فتاوى ورسائل العثيمين": ("ابن عثيمين", "الفتاوى"),
    "فتاوى ابن تيمية": ("ابن تيمية", "الفتاوى"),  # إضافي

    # 📚 متون الحديث (4 كتب)
    "السنن الكبرى للبيهقي ت التركي": ("البيهقي", "متون الحديث"),
    "المحيط في الاحاديث النبوية والسنن والاثار": ("مجموعة مؤلفين", "متون الحديث"),
    "جامع الرويات": ("مجموعة مؤلفين", "متون الحديث"),
    "صحيح البخاري": ("البخاري", "متون الحديث"),

    # 🕌 العقيدة (3 كتب)
    "حلية الأولياء وطبقات الأصفياء": ("أبو نعيم الأصفهاني", "العقيدة"),
    "الجامع لشعب الإيمان للبيهقي": ("البيهقي", "العقيدة"),
    "الموسوعة العقدية - الدرر السنية": ("الدرر السنية", "العقيدة"),

    # 📜 التفاسير (5 كتب أساسية + 2 إضافيين)
    "المهذب النقي الجامع لتفسير ابن جرير الطبري": ("ابن جرير الطبري", "التفاسير"),
    "الموسوعة القرآنية": ("مجموعة مؤلفين", "التفاسير"),
    "تفسير ابن كثير _": ("ابن كثير", "التفاسير"),
    "تفسير القرطبي": ("القرطبي", "التفاسير"),
    "روح البيان": ("الإسماعيلي", "التفاسير"),
    "تفسير الرازي = مفاتيح الغيب": ("الرازي", "التفاسير"),  # إضافي
    "المغني لابن قدامة": ("ابن قدامة", "فقه عام"),  # إضافي (وضعته في فقه عام)
}

BOOK_NAMES = list(BOOKS_DATA.keys())

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

# ==================== دالة التوليد (تم تعديل التنسيق ليطابق التدريب) ====================
@spaces.GPU(duration=120)
def generate_text(prompt, book, author, category, max_new_tokens):
    # ✅ التعديل الجوهري: نفس صيغة format_example في التدريب
    # book X author Y category Z\n(prompt)
    full_prompt = f"book {book} author {author} category {category}\n{prompt}"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            repetition_penalty=1.02,
            no_repeat_ngram_size=2,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return generated

# ==================== دوال تحديث الحقول عند اختيار الكتاب ====================
def update_author_and_category(book_name):
    if book_name in BOOKS_DATA:
        author, category = BOOKS_DATA[book_name]
        return author, category
    return "", ""

# ==================== تصميم واجهة Gradio ====================
with gr.Blocks(title="محرك النصوص التراثية - نسخة تجريبية", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📖 3arabLM-4B | محرك النصوص التراثية")
    gr.Markdown("💬 **التعليقات مرئية للجميع** - ساعدنا في تحسين النموذج بمشاركة رأيك.")

    with gr.Row():
        with gr.Column(scale=1):
            # قائمة منسدلة تحتوي على 25 كتاباً
            book_input = gr.Dropdown(
                choices=BOOK_NAMES,
                label="📚 اختيار الكتاب",
                value=BOOK_NAMES[0]
            )
            author_input = gr.Textbox(
                label="👤 اسم المؤلف",
                value=BOOKS_DATA[BOOK_NAMES[0]][0] if BOOK_NAMES else "",
                interactive=True
            )
            # ✅ التصنيفات الآن عربية ومطابقة لكود التدريب
            category_input = gr.Dropdown(
                choices=CATEGORIES_MAIN,
                label="📂 التصنيف الرئيسي",
                value=BOOKS_DATA[BOOK_NAMES[0]][1] if BOOK_NAMES else CATEGORIES_MAIN[0],
                interactive=True
            )
            max_tokens_slider = gr.Slider(
                minimum=50, maximum=512, value=300, step=50,
                label="📏 الطول الأقصى للنص المولد (Tokens)"
            )

        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="النص / السؤال",
                placeholder="اكتب النص أو السؤال الذي تريد استكماله...",
                lines=8
            )
            generate_btn = gr.Button("🚀 توليد النص", variant="primary")
            output_text = gr.Textbox(label="الناتج", lines=12)

    # ربط اختيار الكتاب بتحديث المؤلف والتصنيف تلقائياً
    book_input.change(
        fn=update_author_and_category,
        inputs=book_input,
        outputs=[author_input, category_input]
    )

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
        inputs=[prompt_input, book_input, author_input, category_input, max_tokens_slider],
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
