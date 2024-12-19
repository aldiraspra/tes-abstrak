import streamlit as st
import requests
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Ambil API Key dan Endpoint dari .env
API_KEY = os.getenv("API_KEY")
ENDPOINT = os.getenv("ENDPOINT")

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# Fungsi untuk memuat referensi dari database
def load_references():
    conn = sqlite3.connect("references.db")
    c = conn.cursor()
    c.execute("SELECT content FROM abstracts")
    references = [row[0] for row in c.fetchall()]
    conn.close()
    return references

# Fungsi untuk mencari referensi yang relevan
def get_relevant_references(user_input, top_k=3):
    references = load_references()
    corpus = references + [user_input]

    # Hitung TF-IDF dan kesamaan kosinus
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Pilih referensi paling relevan
    top_indices = cosine_sim.argsort()[0][-top_k:][::-1]
    relevant_refs = [references[i] for i in top_indices]

    return relevant_refs

# Fungsi untuk menyusun prompt dinamis
def generate_dynamic_prompt(user_input):
    relevant_references = get_relevant_references(user_input)

    # Format referensi untuk prompt
    formatted_references = "\n".join(
        [f"{idx+1}. **Referensi {idx+1}**\n   {ref}" for idx, ref in enumerate(relevant_references)]
    )

    # Prompt sistem
    system_prompt = f"""
    Role:
    Anda adalah model AI yang dirancang khusus untuk memperbaiki abstrak dokumen hak paten. Fokus utama Anda adalah meningkatkan kualitas penulisan abstrak berdasarkan struktur, terminologi teknis, dan gaya penulisan yang khas dalam dokumen paten. Anda tidak diperkenankan menjawab pertanyaan di luar konteks perbaikan abstrak. Jika Anda menerima pertanyaan atau permintaan yang tidak terkait dengan perbaikan abstrak, Anda harus memberikan jawaban: "Saya hanya dapat membantu memperbaiki abstrak dokumen hak paten. Silakan masukkan abstrak untuk diperbaiki."

    ---

    Konteks:
    Pengguna akan memberikan abstrak dokumen paten yang memerlukan perbaikan. Tugas Anda adalah menganalisis abstrak tersebut dan memperbaikinya berdasarkan pola, struktur, dan gaya penulisan yang terdapat dalam Referensi Abstrak Paten. Pastikan abstrak yang dihasilkan:
    - Mempertahankan makna teknis dan legal dari abstrak asli.
    - Menggunakan terminologi teknis yang akurat.
    - Meningkatkan keterbacaan tanpa mengubah informasi penting.
    
    Referensi Abstrak Paten:
    {formatted_references}

    ---

    Proses Perbaikan:
    1. **Analisis Struktur**: Identifikasi elemen-elemen kunci dalam abstrak pengguna.
    2. **Perbaikan Bahasa**: Tingkatkan tata bahasa dan keterbacaan abstrak.
    3. **Konsistensi Terminologi**: Gunakan istilah teknis yang konsisten.
    4. **Penyusunan Ulang**: Susun ulang kalimat untuk meningkatkan alur logis.
    5. **Keakuratan Teknis**: Jangan ubah makna teknis dalam abstrak pengguna.

    ---

    Tugas Anda:
    Perbaiki abstrak pengguna berdasarkan referensi di atas dan pedoman yang diberikan. Jika Anda menerima input yang tidak terkait dengan perbaikan abstrak, beri jawaban: "Saya hanya dapat membantu memperbaiki abstrak dokumen hak paten. Silakan masukkan abstrak untuk diperbaiki."
    """
    return system_prompt

def improve_abstract(user_input):
    dynamic_prompt = generate_dynamic_prompt(user_input)

    payload = {
        "messages": [
            {"role": "system", "content": dynamic_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800
    }

    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        assistant_reply = result['choices'][0]['message']['content']
        return assistant_reply
    except requests.RequestException as e:
        return f"Gagal mendapatkan respons dari API. Error: {e}"

# Inisialisasi session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Streamlit UI
st.set_page_config(page_title="📝 Chatbot Perbaikan Abstrak", layout="wide")

st.title("📝 Chatbot Perbaikan Abstrak")

# Chat container
response_container = st.container()
input_container = st.container()

with input_container:
    user_input = st.text_area("Masukkan Abstrak Anda:", height=100, key="input")
    submit_button = st.button("📤 Kirim")

if submit_button and user_input:
    if user_input.strip() == "":
        st.warning("Silakan masukkan abstrak yang ingin diperbaiki.")
    else:
        with st.spinner("🔄 Memproses..."):
            assistant_reply = improve_abstract(user_input)
        # Simpan ke session state
        st.session_state.past.append(user_input)
        st.session_state.generated.append(assistant_reply)

with response_container:
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
    else:
        st.info("Silakan masukkan abstrak Anda di bawah dan tekan 'Kirim' untuk memulai.")

# Catatan tambahan
st.markdown("---")
st.markdown("**Catatan:** Chatbot ini hanya berfungsi untuk memperbaiki abstrak. Pertanyaan di luar konteks perbaikan abstrak tidak akan dijawab.")
