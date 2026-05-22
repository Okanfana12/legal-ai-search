import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retriever import load_vector_store, answer_question, RAGResponse
from config import ANTHROPIC_MODEL, TOP_K_RESULTS, CONFIDENCE_THRESHOLD

st.set_page_config(
    page_title="Legal AI Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #f8f8f7; border-right: 1px solid #e5e5e3; }
[data-testid="stSidebar"] .stButton > button { width: 100%; text-align: left; border: none; background: transparent; color: #555; border-radius: 8px; padding: 0.5rem 0.75rem; font-size: 0.9rem; }
[data-testid="stSidebar"] .stButton > button:hover { background: #efefed; color: #111; }
.main-title { font-size: 1.6rem; font-weight: 600; margin-bottom: 0.2rem; }
.subtitle { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
.result-box { border-left: 3px solid #1D9E75; padding: 1rem 1.25rem; border-radius: 0 10px 10px 0; background: #f0faf6; margin-top: 1rem; }
.result-box.warn { border-left-color: #BA7517; background: #fef9f0; }
.source-tag { display: inline-block; background: #f1f0e8; color: #555; border-radius: 20px; padding: 2px 10px; font-size: 0.78rem; margin: 3px 3px 0 0; }
.chip { display: inline-block; border: 1px solid #ddd; border-radius: 20px; padding: 2px 10px; font-size: 0.78rem; color: #666; margin: 2px; cursor: pointer; }
div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Legal AI Search")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🔍  Recherche", "🕘  Historique", "📊  Statistiques", "⚙️  Paramètres"],
        label_visibility="collapsed",
    )
    st.divider()
    with st.expander("Modèle actif", expanded=True):
        st.caption(f"**{ANTHROPIC_MODEL}**")
        st.caption(f"Top-K : {TOP_K_RESULTS}  ·  Seuil : {CONFIDENCE_THRESHOLD}")


# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement de la base vectorielle…")
def get_vector_store():
    return load_vector_store()


# ── Pages ─────────────────────────────────────────────────────────────────────

# ── 1. Recherche ─────────────────────────────────────────────────────────────
if page == "🔍  Recherche":
    st.markdown('<p class="main-title">Recherche juridique</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Posez votre question en langage naturel</p>', unsafe_allow_html=True)

    try:
        vector_store = get_vector_store()
    except Exception as e:
        st.error(f"Impossible de charger ChromaDB : {e}")
        st.stop()

    # Quick suggestions
    suggestions = [
        "Licenciement économique",
        "Préavis de démission",
        "Harcèlement au travail",
        "Rupture conventionnelle",
    ]
    cols = st.columns(len(suggestions))
    prefill = None
    for col, sug in zip(cols, suggestions):
        if col.button(sug, use_container_width=True):
            prefill = sug

    question = st.text_area(
        "Votre question",
        value=prefill or st.session_state.get("prefill_q", ""),
        placeholder="Ex : Quelles sont les conditions de rupture d'un contrat de travail ?",
        height=120,
        label_visibility="collapsed",
    )

    col_btn, col_clear = st.columns([1, 5])
    submitted = col_btn.button("🔍 Analyser", type="primary", use_container_width=True)
    if col_clear.button("Effacer", use_container_width=False):
        st.session_state["prefill_q"] = ""
        st.rerun()

    if submitted:
        if not question.strip():
            st.warning("Veuillez saisir une question.")
        else:
            st.session_state.setdefault("history", [])
            with st.spinner("Analyse en cours…"):
                response: RAGResponse = answer_question(vector_store, question.strip())

            # Save to history
            st.session_state["history"].append({
                "question": question.strip(),
                "confidence": response.confidence,
                "reliable": response.is_reliable,
                "answer": response.answer,
                "sources": response.sources,
            })

            conf_pct = f"{response.confidence * 100:.1f}%"
            if response.is_reliable:
                st.markdown(
                    f'<div class="result-box">'
                    f'<p style="color:#0F6E56;font-weight:600;margin-bottom:.5rem">✅ Confiance : {conf_pct}</p>'
                    f'{response.answer}</div>',
                    unsafe_allow_html=True,
                )
                if response.sources:
                    st.markdown("**Sources**")
                    for src in response.sources:
                        st.markdown(f'<span class="source-tag">📄 {src}</span>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="result-box warn">'
                    f'<p style="color:#854F0B;font-weight:600;margin-bottom:.5rem">⚠️ Confiance insuffisante ({conf_pct})</p>'
                    f'{response.answer}</div>',
                    unsafe_allow_html=True,
                )


# ── 2. Historique ─────────────────────────────────────────────────────────────
elif page == "🕘  Historique":
    st.markdown('<p class="main-title">Historique des recherches</p>', unsafe_allow_html=True)
    history = st.session_state.get("history", [])
    if not history:
        st.info("Aucune recherche effectuée pour l'instant.")
    else:
        for i, item in enumerate(reversed(history)):
            conf_pct = f"{item['confidence'] * 100:.1f}%"
            badge = "✅ Fiable" if item["reliable"] else "⚠️ Faible"
            with st.expander(f"{badge} · {item['question'][:80]}… · {conf_pct}"):
                st.write(item["answer"])
                if item["sources"]:
                    for src in item["sources"]:
                        st.markdown(f'<span class="source-tag">📄 {src}</span>', unsafe_allow_html=True)
        if st.button("🗑️ Vider l'historique"):
            st.session_state["history"] = []
            st.rerun()


# ── 3. Statistiques ──────────────────────────────────────────────────────────
elif page == "📊  Statistiques":
    st.markdown('<p class="main-title">Statistiques</p>', unsafe_allow_html=True)
    history = st.session_state.get("history", [])
    total = len(history)
    reliable = sum(1 for h in history if h["reliable"])
    avg_conf = (sum(h["confidence"] for h in history) / total * 100) if total else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Requêtes traitées", total)
    c2.metric("Réponses fiables", reliable)
    c3.metric("Confiance moyenne", f"{avg_conf:.1f}%")

    if history:
        st.divider()
        st.subheader("Détail des requêtes")
        import pandas as pd
        df = pd.DataFrame([{
            "Question": h["question"][:60] + "…",
            "Confiance": f"{h['confidence']*100:.1f}%",
            "Fiable": "✅" if h["reliable"] else "⚠️",
        } for h in reversed(history)])
        st.dataframe(df, use_container_width=True, hide_index=True)


# ── 4. Paramètres ────────────────────────────────────────────────────────────
elif page == "⚙️  Paramètres":
    st.markdown('<p class="main-title">Paramètres</p>', unsafe_allow_html=True)

    with st.form("settings_form"):
        st.subheader("Moteur RAG")
        model = st.selectbox("Modèle IA", [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ], index=0)
        top_k = st.slider("Top-K résultats", 1, 20, TOP_K_RESULTS)
        threshold = st.slider("Seuil de confiance", 0.0, 1.0, float(CONFIDENCE_THRESHOLD), step=0.05)

        st.subheader("Affichage")
        show_sources = st.toggle("Afficher les sources", value=True)
        strict_mode = st.toggle("Mode strict (refuser sous le seuil)", value=False)

        if st.form_submit_button("💾 Sauvegarder", type="primary"):
            st.success("Paramètres sauvegardés. Redémarrez l'app pour appliquer les changements.")
            st.session_state["settings"] = {
                "model": model,
                "top_k": top_k,
                "threshold": threshold,
                "show_sources": show_sources,
                "strict_mode": strict_mode,
            }