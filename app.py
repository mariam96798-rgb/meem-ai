
import os
import numpy as np
import pandas as pd
import streamlit as st

# OpenAI client
from typing import List, Dict
try:
    from openai import OpenAI
    client = OpenAI()
except Exception:
    client = None

APP_TITLE = "Meem AI — دردشة مبنية على بوستات مريم"

WELCOME_AR = (
    "مرحبًا 👋\n"
    "**Meem AI** النسخة التفاعلية من محتوى مريم.\n"
    "هنا تقدر تسألني عن **التسويق الرقمي، النمو، التفكير الإبداعي، والكتب التي تحسّن جودة الحياة**.\n"
    "كل إجاباتي مستوحاة من منشورات مريم على لينكدإن — بأسلوب بسيط ومليان عمق. 💛\n"
    "اسأل براحتك، أحيانًا كلمة تغيّر نظرتك. 🌱"
)
WELCOME_EN = (
    "Hi there 👋\n"
    "I’m **Meem AI**, the interactive version of Maryam’s content.\n"
    "Ask me about **digital marketing, growth, creative thinking, and books that improve quality of life**.\n"
    "All answers are inspired by Maryam’s original LinkedIn posts — in her simple yet deep style. 💛\n"
    "Feel free to ask anything; one word can shift your perspective. 🌱"
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
MEEM_TOKEN = os.getenv("MEEM_TOKEN", "")  # optional token gate

# ----------------------- Data -----------------------
@st.cache_data(show_spinner=False)
def load_posts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    required = {"date","title","url","content"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    # Drop empty content rows
    df = df[df["content"].astype(str).str.strip()!=""].reset_index(drop=True)
    return df

def embed_texts(texts: List[str]) -> np.ndarray:
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Install openai and set OPENAI_API_KEY.")
    out = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs = [np.array(e.embedding, dtype=np.float32) for e in resp.data]
        out.append(np.vstack(vecs))
    return np.vstack(out) if out else np.zeros((0,1536), dtype=np.float32)

def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

@st.cache_resource(show_spinner=False)
def build_index(df: pd.DataFrame) -> np.ndarray:
    return embed_texts(df["content"].astype(str).tolist())

def retrieve(df: pd.DataFrame, mat: np.ndarray, query: str, k: int=4) -> List[int]:
    if len(df)==0:
        return []
    qv = embed_texts([query])
    sims = (_norm(qv) @ _norm(mat).T).ravel()
    order = np.argsort(-sims)[:k]
    return order.tolist()

def chat_answer(query: str, contexts: List[Dict[str,str]]) -> str:
    if client is None:
        return "⚠️ لم يتم تهيئة OpenAI. تأكدي من تثبيت الحزمة وضبط OPENAI_API_KEY."
    ctx_txt = "\n\n".join([
        f"[{i+1}] {c['title']} — {c['date']}\n{c['content'][:900]}"
        for i,c in enumerate(contexts)
    ])
    system_msg = (
        "أنت مساعد يجيب بالعربية الفصحى وبأسلوب مريم الهادئ والمباشر، "
        "وتضيف سطرًا إنجليزيًا مختصرًا *عند الحاجة فقط*. "
        "اعتمد حصريًا على المقاطع المرجعية التالية من منشورات مريم. "
        "إن لم تجد إجابة كافية، قل ذلك واقترح أقرب منشور ذي صلة. "
        "لا تستخدم أي معلومات من الويب."
    )
    user_msg = (
        f"سؤال المستخدم: {query}\n\n"
        f"المقاطع المرجعية:\n{ctx_txt}\n\n"
        "أجب بإيجاز (٥–٨ أسطر). إذا وُجدت روابط ضمن المقاطع، اذكرها بإيجاز."
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":system_msg},
                  {"role":"user","content":user_msg}],
        temperature=0.3,
        max_tokens=500,
    )
    return resp.choices[0].message.content

# ----------------------- UI -----------------------
st.set_page_config(page_title="Meem AI", page_icon="💛", layout="centered")

# Token gate (optional)
if MEEM_TOKEN:
    with st.sidebar:
        st.markdown("### 🔒 رابط خاص")
        t = st.text_input("أدخل رمز الوصول (Token):", type="password")
        if t != MEEM_TOKEN:
            st.info("هذا الشات خاص — أدخل الرمز الصحيح للوصول.")
            st.stop()

# Logo & headers
col1, col2 = st.columns([1,3])
with col1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
with col2:
    st.markdown(f"## {APP_TITLE}")
st.write(WELCOME_AR)
st.write("---")
st.write(WELCOME_EN)
st.write("---")

# Data source block
with st.expander("📄 مصدر البيانات (Posts CSV)", expanded=False):
    st.caption("ملف posts.csv مرفق داخل التطبيق. يمكنك رفع ملف جديد لاستبداله مؤقتًا.")
    uploaded = st.file_uploader("ارفع ملف CSV مطابق للأعمدة: date, title, url, content", type=["csv"])

csv_path = "posts.csv"
if uploaded is not None:
    csv_path = uploaded

try:
    df = load_posts(csv_path)
except Exception as e:
    st.error(f"خطأ في قراءة CSV: {e}")
    st.stop()

try:
    index = build_index(df)
except Exception as e:
    st.error(f"خطأ في إنشاء الفهرس (Embeddings): {e}")
    st.stop()

# Chat box
st.markdown("### 💬 اسأل Meem AI")
q = st.text_input("اكتب سؤالك هنا...", placeholder="مثال: كيف أطبق قاعدة الدقيقتين؟ أو أعطني كتابًا يحسّن جودة حياتي.")
topk = st.slider("عدد المقاطع المرجعية", min_value=3, max_value=6, value=4)

if st.button("إرسال", use_container_width=True) or (q and st.session_state.get("enter_pressed")):
    if not q.strip():
        st.warning("اكتب سؤالك أولًا.")
    else:
        idxs = retrieve(df, index, q, k=topk)
        ctxs = [{
            "date": df.iloc[i]["date"],
            "title": df.iloc[i]["title"],
            "url": df.iloc[i]["url"],
            "content": df.iloc[i]["content"],
        } for i in idxs]
        with st.spinner("يتم توليد الإجابة…"):
            ans = chat_answer(q, ctxs)
        st.markdown("#### ✨ الإجابة")
        st.write(ans)
        st.markdown("#### 🔎 المقاطع المرجعية")
        for i, c in enumerate(ctxs, 1):
            st.markdown(f"**[{i}] {c['title']} — {c['date']}**")
            if c["url"]:
                st.write(f"🔗 {c['url']}")
            st.write(c["content"][:600] + ("…" if len(c["content"])>600 else ""))
            st.write("---")

st.caption("© Meem AI — مبني على منشورات مريم فقط. لا مصادر خارجية.")
