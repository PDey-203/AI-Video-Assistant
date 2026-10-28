from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

from core.summarise import summarize_source

load_dotenv()

llm = ChatOpenRouter(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    temperature=0.2,
    max_retries=2,
)


EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured information from summaries of audio/video "
            "transcripts. Only use what is written in the text. Never invent "
            "tasks, owners, deadlines, questions or decisions.",
        ),
        (
            "human",
            "From the summary below, extract:\n"
            "1. Action items: concrete tasks someone should do "
            "(include owner and deadline only if stated).\n"
            "2. Decisions: conclusions or choices that were made.\n\n"
            "Use exactly this format:\n"
            "ACTION ITEMS\n- item\n\n"
            "DECISIONS\n- item\n\n"
            "If a section has nothing, write '- None found' under it. "
            "Reply with this format only, with no extra text.\n\n"
            "Summary:\n{summary}",
        ),
    ]
)

chain = EXTRACT_PROMPT | llm | StrOutputParser()


def extract_info(summary):

    if not summary or not summary.strip():
        raise ValueError("Summary text is empty.")

    return chain.invoke({"summary": summary}).strip()


def extract_from_summary(source, language="en", translate=False):

    summary = summarize_source(source, language=language, translate=translate)
    return extract_info(summary)
