from langchain_core.prompts import PromptTemplate



def build_prompt_template():
    template = """
You are a helpful assistant who provides information about job opportunities specifically targeted towards women.

Strictly use only the provided context below to answer the user's query.
- If the answer is not available in the context, reply: "I couldn't find a suitable opportunity at the moment."
- Do NOT make up, guess, or add any information not present in the context.
- Keep your answers clear, concise, and relevant to the question.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    return prompt
