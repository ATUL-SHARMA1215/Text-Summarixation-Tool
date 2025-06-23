import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Page settings
st.set_page_config(page_title="AI Text Summarization Tool", layout="centered", page_icon="🧠")

# CSS Styling
st.markdown("""
<style>
.stApp {
    background-image: url("https://miro.medium.com/v2/resize:fit:1200/1*N_rpqtlvyepXP01EuxbGmw.jpeg");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    color: white;
}
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: white !important;
}
textarea, input[type="text"], .stTextArea, .stTextInput {
    background-color: rgba(0, 0, 0, 0.6) !important;
    color: #ffffff !important;
    border: 2px solid #00cccc;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
}
.stButton > button {
    background-color: #00cccc;
    color: white !important;
    border: none;
    border-radius: 10px;
    font-size: 18px;
    padding: 10px 20px;
    font-weight: bold;
    transition: 0.3s ease;
}
.stButton > button:hover {
    background-color: #009999;
    transform: scale(1.05);
}
.box-style {
    background-color: rgba(0, 0, 0, 0.6);
    padding: 20px;
    border-radius: 12px;
    border: 2px solid #00cccc;
    color: #ffffff !important;
    font-size: 17px;
    white-space: pre-wrap;
    margin-top: 20px;
    text-align: center;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

# Header Box with narrower width
st.markdown("""
<div class="box-style">
    <h1>🧠 AI Text Summarization Tool</h1>
    <p>Summarize long content into short, insightful statements using Facebook’s BART model.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Load model
@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Input method
st.subheader("📥 Input Text")
input_method = st.radio("Choose input method:", ["✍️ Type Text", "📄 Upload .txt File"])
text = ""

if input_method.startswith("✍️"):
    text = st.text_area("Enter your text here:", height=250, placeholder="Paste your content here...")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.markdown(f'<div class="box-style">{text}</div>', unsafe_allow_html=True)

# Word count
if text:
    original_word_count = len(text.split())
    st.markdown(f'<div class="box-style">📝 <strong>Original Word Count:</strong> {original_word_count}</div>', unsafe_allow_html=True)

# Summary settings
with st.expander("⚙️ Summary Settings"):
    min_len = st.slider("Minimum summary length", 10, 100, 30)
    max_len = st.slider("Maximum summary length", 50, 300, 130)

# Generate summary
summary = ""
if st.button("🚀 Generate Summary"):
    if text.strip():
        with st.spinner("Generating summary..."):
            inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=384).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
                summary_ids = model.generate(
                    inputs.input_ids,
                    num_beams=4,
                    min_length=min_len,
                    max_length=max_len,
                    early_stopping=True
                )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.success("✅ Summary Generated")
        st.markdown("### 🧾 Summary")
        st.markdown(f'<div class="box-style">{summary}</div>', unsafe_allow_html=True)

        # Stats
        summary_word_count = len(summary.split())
        compression_ratio = round(summary_word_count / original_word_count * 100, 2)
        stats_html = f"""
        <div class="box-style">
        🔹 <strong>Original Words:</strong> {original_word_count}<br>
        🔹 <strong>Summary Words:</strong> {summary_word_count}<br>
        🔹 <strong>Compression Ratio:</strong> {compression_ratio}% of original
        </div>
        """
        st.markdown("<h4>📊 Summary Stats:</h4>", unsafe_allow_html=True)
        st.markdown(stats_html, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter or upload some text.")

# Footer
st.markdown("---")
st.caption("Made with ❤️by Atul Sharma")
