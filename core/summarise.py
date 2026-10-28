from core.transcriber import transcribe_source
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenRouter(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    temperature=0.2,
    max_retries=2,
)


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert summarizer. You work from audio/video transcripts "
            "that may contain filler words, repetition, or transcription errors. "
            "Be accurate and never invent facts that are not in the text.",
        ),
        (
            "human",
            "Write a clear, concise summary of the text below as plain paragraphs. "
            "If the text is made of several partial summaries, merge them into one "
            "coherent summary without repeating points. Reply with the summary only.\n\n"
            "Text:\n{text}",
        ),
    ]
)


chains = SUMMARY_PROMPT | llm | StrOutputParser()


def summarize_text(
    text,
    chunk_size=3000,
    chunk_overlap=500,
    max_concurrency=4,
):

    if not text or not text.strip():
        raise ValueError("Transcript text is empty.")

    if len(text) <= chunk_size:
        return chains.invoke({"text": text})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    map_inputs = [
        {"text": chunk, "part": i, "total": len(chunks)}
        for i, chunk in enumerate(chunks, 1)
    ]

    partial_summaries = chains.batch(
        map_inputs, config={"max_concurrency": max_concurrency}
    )

    combined = "\n\n".join(
        f"[Part {i}]\n{s}" for i, s in enumerate(partial_summaries, 1)
    )

    return chains.invoke({"text": combined})


def summarize_source(source, language="en", translate=False):

    transcript = transcribe_source(source, language=language, translate=translate)
    return summarize_text(transcript)
