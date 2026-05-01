import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retriever import load_vector_store, answer_question, RAGResponse
from config import ANTHROPIC_MODEL, TOP_K_RESULTS, CONFIDENCE_THRESHOLD

st.set_page_config(
    page_title="Legal AI Search",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Legal AI Search")
st.caption(f"Modèle : `{ANTHROPIC_MODEL}` · Top-K : {TOP_K_RESULTS} · Seuil confiance : {CONFIDENCE_THRESHOLD}")


@st.cache_resource(show_spinner="Chargement de la base vectorielle…")
def get_vector_store():
    return load_vector_store()


def render_response(response: RAGResponse) -> None:
    confidence_pct = f"{response.confidence * 100:.1f}%"

    if response.is_reliable:
        st.success(f"Confiance : {confidence_pct}")
        st.markdown(response.answer)

        if response.sources:
            with st.expander("Sources"):
                for src in response.sources:
                    st.write(f"- {src}")
    else:
        st.warning(f"Confiance insuffisante ({confidence_pct})")
        st.info(response.answer)


try:
    vector_store = get_vector_store()
except Exception as e:
    st.error(f"Impossible de charger ChromaDB : {e}")
    st.stop()

with st.form("question_form", clear_on_submit=False):
    question = st.text_area(
        "Votre question juridique",
        placeholder="Ex : Quelles sont les conditions de rupture d'un contrat de travail ?",
        height=100
    )
    submitted = st.form_submit_button("Rechercher", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Veuillez saisir une question.")
    else:
        with st.spinner("Analyse en cours…"):
            response = answer_question(vector_store, question.strip())
        render_response(response)
