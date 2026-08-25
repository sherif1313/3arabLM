#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار استرجاع النصوص (Faithful Recall) من نموذج Qwen المُدرَّب على المكتبة الشاملة.
يتوافق تنسيق الـ Prompt مع كود التدريب تماماً.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 🔧 الإعدادات
# ============================================================

MODEL_PATH = "sherif1313/3arabLM-4B-islamic-v2"  # غيّره إلى مسار نموذجك
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 384

# ============================================================
# 📦 تحميل النموذج والمحلل اللغوي
# ============================================================

print("⏳ تحميل النموذج...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ============================================================
# 📝 دالة تنسيق الميتاداتا (مطابقة تماماً للتدريب)
# ============================================================

def format_prefix(book=None, author=None, category=None, part=None, page=None, hierarchy=None):
    """
    توليد الميتاداتا بنفس تنسيق التدريب:
    - جميع الحقول في سطر واحد، مفصولة بمسافات
    - بدون نقطتين (:) بعد اسم الحقل
    - part> و page> و title> (مع >)
    - hierarchy تؤخذ منها القيمة الأخيرة فقط
    """
    tags = []

    if book:
        tags.append(f"book {book}")
    if author:
        tags.append(f"author {author}")
    if category:
        tags.append(f"category {category}")
    if part is not None and str(part).strip():
        tags.append(f"part> {part}")
    if page is not None and str(page).strip():
        tags.append(f"page> {page}")
    if hierarchy:
        parts = hierarchy.split('|')
        title = parts[-1] if parts else ''
        if title:
            tags.append(f"title> {title}")

    return " ".join(tags)

# ============================================================
# 🤖 دالة توليد الإجابة (استرجاع أمين)
# ============================================================

def generate_text(prompt, book=None, author=None, category=None, part=None, page=None, hierarchy=None):
    # بناء الـ prompt بالشكل: "metadata\nالنص"
    prefix = format_prefix(book, author, category, part, page, hierarchy)
    full_prompt = f"{prefix}\n{prompt}" if prefix else prompt

    print(f"\n📝 السؤال/النص: {prompt}")
    print("-" * 50)

    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,                # استرجاع حتمي (Greedy)
            repetition_penalty=1.02,
            no_repeat_ngram_size=2,
            #repetition_penalty=None
            #no_repeat_ngram_size=None

           
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"🤖 إجابة النموذج:\n{answer}")
    print("=" * 80)

# ============================================================
# 📚 أسئلة تستند إلى الكتب الموجودة فعلاً في ./1/2
# ============================================================

TEST_CASES = [

    # ================================================
    # الفقه
    # ========================================================

    {
        "prompt": " حكم المسح على الخفين؟",
        "book": "الموسوعة الفقهية الكويتية",
        "author": "وزارة الأوقاف والشؤون الإسلامية بالكويت",
        "category": "فقه عام"
    },

    {
        "prompt": " شروط الصلاة؟",
        "book": "الموسوعة الفقهية الكويتية",
        "author": "وزارة الأوقاف والشؤون الإسلامية بالكويت",
        "category": "فقه عام"
    },

    {
        "prompt": " حكم صلاة المسافر؟",
        "book": "الموسوعة الفقهية الكويتية",
        "author": "وزارة الأوقاف والشؤون الإسلامية بالكويت",
        "category": "فقه عام"
    },

    {
        "prompt": " حكم الجمع بين الصلاتين في السفر؟",
        "book": "الموسوعة الفقهية الكويتية",
        "author": "وزارة الأوقاف والشؤون الإسلامية بالكويت",
        "category": "فقه عام"
    },

    {
        "prompt": " أركان الصلاة؟",
        "book": "موسوعة فقه العبادات",
        "author": "علي بن نايف الشحود",
        "category": "فقه عام"
    },

    {
        "prompt": " سنن الصلاة؟",
        "book": "موسوعة فقه العبادات",
        "author": "علي بن نايف الشحود",
        "category": "فقه عام"
    },

    {
        "prompt": " مبطلات الوضوء؟",
        "book": "موسوعة فقه العبادات",
        "author": "علي بن نايف الشحود",
        "category": "فقه عام"
    },

    {
        "prompt": " حكم زكاة الحلي؟",
        "book": "مجموع فتاوى ورسائل العثيمين",
        "author": "محمد بن صالح العثيمين",
        "category": "الفتاوى"
    },

    {
        "prompt": " حكم الجمع والقصر في السفر؟",
        "book": "فتاوى الشبكة الإسلامية",
        "author": "لجنة الفتوى بالشبكة الإسلامية",
        "category": "الفتاوى"
    },


    # ========================================================
    # الحديث والآثار
    # ========================================================

    {
        "prompt": " أول حديث في صحيح البخاري؟",
        "book": "صحيح البخاري - ط الشعب",
        "author": "محمد بن إسماعيل البخاري",
        "category": "متون الحديث"
    },

    {
        "prompt": " حكم الأعمال بالنيات؟",
        "book": "صحيح البخاري - ط الشعب",
        "author": "محمد بن إسماعيل البخاري",
        "category": "متون الحديث"
    },

    {
        "prompt": " حديث جبريل في الإسلام والإيمان والإحسان؟",
        "book": "صحيح البخاري - ط الشعب",
        "author": "محمد بن إسماعيل البخاري",
        "category": "متون الحديث"
    },

    {
        "prompt": " فضل طلب العلم؟",
        "book": "الجامع لشعب الإيمان للبيهقي",
        "author": "أحمد بن الحسين البيهقي",
        "category": "متون الحديث"
    },

    {
        "prompt": " فضل الصدق؟",
        "book": "السنن الكبرى للبيهقي ت التركي",
        "author": "أحمد بن الحسين البيهقي",
        "category": "متون الحديث"
    },

    {
        "prompt": " فضل ذكر الله تعالى؟",
        "book": "المحيط في الاحاديث النبوية والسنن والاثار",
        "author": "غير محدد",
        "category": "متون الحديث"
    },

    {
        "prompt": " فضل الصلاة على النبي صلى الله عليه وسلم؟",
        "book": "جامع الرويات",
        "author": "غير محدد",
        "category": "متون الحديث"
    },

    {
        "prompt": " منزلة الصدق في سيرة الصالحين؟",
        "book": "حلية الأولياء وطبقات الأصفياء",
        "author": "أبو نعيم الأصبهاني",
        "category": "متون الحديث"
    },


    # ========================================================
    # العقيدة
    # ========================================================

    {
        "prompt": " معنى توحيد الله تعالى؟",
        "book": "الموسوعة العقدية - الدرر السنية",
        "author": "الموسوعة العقدية",
        "category": "العقيدة"
    },

    {
        "prompt": " أقسام التوحيد؟",
        "book": "الموسوعة العقدية - الدرر السنية",
        "author": "الموسوعة العقدية",
        "category": "العقيدة"
    },

    {
        "prompt": " معنى الإيمان بالله؟",
        "book": "الموسوعة العقدية - الدرر السنية",
        "author": "الموسوعة العقدية",
        "category": "العقيدة"
    },


    # ========================================================
    # التفسير
    # ========================================================

    {
        "prompt": " تفسير سورة الفاتحة؟",
        "book": "تفسير القرطبي",
        "author": "محمد بن أحمد القرطبي",
        "category": "التفاسير"
    },

    {
        "prompt": " معنى قوله تعالى: الحمد لله رب العالمين؟",
        "book": "تفسير الرازي = مفاتيح الغيب أو التفسير الكبير",
        "author": "فخر الدين الرازي",
        "category": "التفاسير"
    },

    {
        "prompt": " تفسير قوله تعالى: قل هو الله أحد؟",
        "book": "روح البيان",
        "author": "إسماعيل حقي",
        "category": "التفاسير"
    },
  
    {
        "prompt": " تفسير قوله تعالى: قل هو الله أحد؟",
        "book": "تفسير ابن كثير",
        "author": "ابن كثير",
        "category": "التفاسير"
    },
  
    {
        "prompt": " معنى التقوى في القرآن الكريم؟",
        "book": "الموسوعة القرآنية",
        "author": "غير محدد",
        "category": "التفاسير"
    },

]


# ============================================================
# 🚀 تشغيل الاختبار
# ============================================================

print("\n🚀 بدء اختبار النموذج على أسئلة متنوعة...\n")

for test in TEST_CASES:
    generate_text(
        prompt=test["prompt"],
        book=test.get("book"),
        author=test.get("author"),
        category=test.get("category"),
        part=test.get("part"),
        page=test.get("page"),
        hierarchy=test.get("hierarchy")
    )

print("\n✅ انتهى الاختبار.")
