from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openrouter import ChatOpenRouter
from core.vector_store import get_retriever

llm = ChatOpenRouter(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    temperature=0.2,
    max_retries=2,
)

retriever = get_retriever(k=4)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using ONLY the transcript context below. "
            "If the answer is not in the context, say you could not find it.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm | StrOutputParser()

history = []


def ask(question):
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    answer = ""
    print("AI: ", end="", flush=True)
    for token in chain.stream(
        {"context": context, "history": history[-10:], "question": question}
    ):
        print(token, end="", flush=True)
        answer += token
    print()

    sources = sorted(
        {f"{d.metadata['source']} (chunk {d.metadata['chunk']})" for d in docs}
    )
    print("Sources:", ", ".join(sources), "\n")

    history.append(HumanMessage(question))
    history.append(AIMessage(answer))
