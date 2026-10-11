"""RAG Playground chat UI.  Run with:  uv run ui"""

import logging
import os

import streamlit as st

from rag import config, embeddings, store
from rag.answer import list_prompts, load_system_prompt, stream_answer
from rag.ingest import sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-request Hugging Face HEAD checks
log = logging.getLogger("rag.app")

st.set_page_config(page_title="RAG Playground", page_icon=":material/search:", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources": [dict]}]


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            jev = f" · Jev relevance {s['rerank']:.1f}/3" if s.get("rerank") is not None else ""
            st.markdown(f"**[{i}] {s['label']}** · similarity {s['score']:.2f}{jev}")
            st.markdown("> " + s["text"].replace("\n", "\n> "))


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.header("System prompt")
    prompts = list_prompts()
    names = [p.name for p in prompts]
    default = names.index(config.SYSTEM_PROMPT_PATH.name) if config.SYSTEM_PROMPT_PATH.name in names else 0
    prompt_name = st.selectbox("Active prompt", names, index=default, label_visibility="collapsed")
    prompt_path = config.PROMPTS_DIR / prompt_name
    with st.expander("View prompt"):
        st.markdown(load_system_prompt(prompt_path))
    st.caption(f"Edit `prompts/{prompt_name}` - changes apply on the next question.")

    st.header("Corpus")
    manifest = store.load_manifest()
    c1, c2 = st.columns(2)
    c1.metric("Files", len(manifest["files"]))
    c2.metric("Chunks", sum(f["chunks"] for f in manifest["files"].values()))
    st.caption(f"Last sync: {manifest['last_sync'] or 'never'}")
    if manifest["files"]:
        with st.expander("Indexed files"):
            for name, info in sorted(manifest["files"].items()):
                st.markdown(f"- `{name}` ({info['chunks']} chunks)")
    rebuild = st.checkbox("Full rebuild", help="Wipe the index and re-embed every file.")
    if st.button("Sync corpus folder", width="stretch", type="primary"):
        with st.status("Syncing corpus...", expanded=True) as status:
            lines: list[str] = []
            stats = sync(rebuild=rebuild, log=lines.append)
            st.code("\n".join(lines) or "Nothing to do.", language=None)
            status.update(label="Sync failed for some files" if stats["failed"] else "Sync complete", state="error" if stats["failed"] else "complete")
        st.button("Refresh stats", width="stretch")

    st.header("Settings")
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true" and os.getenv("LANGSMITH_API_KEY")
    st.caption(
        f"LLM: `{config.LLM_MODEL}`  \nEmbeddings: `{config.EMBED_MODEL}`  \n"
        f"Top-k: {config.TOP_K} · Reranker: "
        f"{f'Jev (top {config.RERANK_CANDIDATES})' if config.RERANKER == 'jev' else 'off'}  \n"
        f"LangSmith tracing: {'on' if tracing else 'off'}"
    )
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# --- chat -------------------------------------------------------------------

st.title("RAG Playground")
st.caption("Ask questions about the documents in `corpus/`.")

if not config.OPENROUTER_API_KEY:
    st.warning("`OPENROUTER_API_KEY` is not set. Add it to `.env` and restart the app.")
if store.collection().count() == 0:
    st.info("The index is empty. Put files in `corpus/` and click **Sync corpus folder** (or run `uv run ingest`).")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def _load_embedder() -> bool:
    embeddings.warm_up()
    return True


_load_embedder()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))

if question := st.chat_input("Ask a question about your corpus"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            events = stream_answer(question, history=history, prompt_path=prompt_path)
            with st.spinner("Searching the corpus..."):
                chunks = next(events)
            text = st.write_stream(events)
            sources = [{"label": c.label, "text": c.text, "score": c.score, "rerank": c.rerank_score} for c in chunks]
            render_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources})
        except Exception as exc:
            log.exception("Failed to answer: %r", question)
            st.error(f"{type(exc).__name__}: {exc}")
