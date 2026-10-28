import os
import tempfile

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from core.transcriber import transcribe_source
from core.summarise import summarize_text
from core.extractor import extract_info
from core.vector_store import add_transcript
from core.rag_pipeline import chain, retriever

st.set_page_config(page_title="AI Assistant", page_icon="🎬", layout="wide")


LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Urdu": "ur",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}


if "source" not in st.session_state:
    st.session_state.source = None
    st.session_state.transcript = None
    st.session_state.summary = None
    st.session_state.extracted = None
    st.session_state.chunks = 0
    st.session_state.messages = []


# ---------- Loading a video ----------
def load_video(source_path, source_name, language, translate):
    """Transcribe the video, save it for chat, and remember it."""
    with st.spinner("Working... this can take a few minutes."):
        transcript = transcribe_source(
            source_path, language=language, translate=translate
        )
        chunks = add_transcript(transcript, source_name)

    st.session_state.source = source_name
    st.session_state.transcript = transcript
    st.session_state.chunks = chunks
    st.session_state.summary = None
    st.session_state.extracted = None
    st.success("Loaded!")


def load_uploaded_file(uploaded, language, translate):
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        temp_path = tmp.name

    try:
        load_video(temp_path, uploaded.name, language, translate)
    finally:
        os.remove(temp_path)


# ---------- Pages ----------
def get_summary():
    if st.session_state.summary is None:
        with st.spinner("Summarising..."):
            st.session_state.summary = summarize_text(st.session_state.transcript)
    return st.session_state.summary


def show_summary():
    st.title("Summary")
    summary = get_summary()
    st.write(summary)
    st.download_button("Download summary", summary, file_name="summary.txt")


def show_action_items():
    st.title("Action items & decisions")

    if st.session_state.extracted is None:
        summary = get_summary()  # the extractor works from the summary
        with st.spinner("Extracting..."):
            st.session_state.extracted = extract_info(summary)

    st.markdown(st.session_state.extracted)
    st.download_button(
        "Download", st.session_state.extracted, file_name="action_items.txt"
    )


def show_transcript():
    st.title("Transcript")
    st.text_area(
        "Full transcript",
        st.session_state.transcript,
        height=500,
        label_visibility="collapsed",
    )
    st.download_button(
        "Download transcript", st.session_state.transcript, file_name="transcript.txt"
    )


def show_sources(sources):
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.write(f"- {s}")


def show_chat():
    st.title("Chat")
    st.caption("Ask questions about everything you've indexed so far.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            show_sources(m.get("sources"))

    question = st.chat_input("Ask a question about your videos...")
    if not question:
        return

    history = []
    for m in st.session_state.messages[-10:]:
        if m["role"] == "user":
            history.append(HumanMessage(m["content"]))
        else:
            history.append(AIMessage(m["content"]))

    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    sources = sorted(
        {f"{d.metadata['source']} (chunk {d.metadata['chunk']})" for d in docs}
    )

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                chain.stream(
                    {"context": context, "history": history, "question": question}
                )
            )
            show_sources(sources)
        except Exception as e:
            answer = f"Error: {e}"
            sources = []
            st.error(answer)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🎬 AI Assistant")
    st.caption("Transcribe, summarise and chat with your videos.")

    st.subheader("1. Load a video / audio")
    mode = st.radio("Input type", ["URL", "Upload file"], horizontal=True)

    if mode == "URL":
        url = st.text_input("Video / audio URL")
    else:
        uploaded = st.file_uploader(
            "Audio or video file",
            type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "mov", "ogg", "flac"],
        )

    language_name = st.selectbox("Spoken language", list(LANGUAGES.keys()))
    translate = st.checkbox("Translate to English")
    language = LANGUAGES[language_name]

    if st.button("Transcribe & index", type="primary", use_container_width=True):
        try:
            if mode == "URL" and url.strip():
                load_video(url.strip(), url.strip(), language, translate)
            elif mode == "Upload file" and uploaded is not None:
                load_uploaded_file(uploaded, language, translate)
            else:
                st.warning("Please provide a URL or upload a file.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    if st.session_state.source:
        st.success(f"Loaded: {st.session_state.source}")
        st.caption(f"{st.session_state.chunks} chunks indexed")

    st.subheader("2. Choose a view")
    page = st.radio(
        "View",
        ["Summary", "Action items & decisions", "Transcript", "Chat"],
        label_visibility="collapsed",
    )

    if st.session_state.messages and st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- Main area ----------
if page == "Chat":
    show_chat()
elif not st.session_state.transcript:
    st.title(page)
    st.info("👈 Load a video or audio from the sidebar first.")
elif page == "Summary":
    show_summary()
elif page == "Action items & decisions":
    show_action_items()
else:
    show_transcript()
