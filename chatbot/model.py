# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from huggingface_hub import login
# from langchain_community.llms import HuggingFacePipeline


# def load_hf_model(model_name="gpt2", token=None):
#     if token:
#         login(token=token)

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype="auto",  # it will use float32 on CPU
#     )

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#         device=-1,  # CPU
#     )

#     llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0})
#     return llm


import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import getpass

load_dotenv(".env")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")


def load_groq():
    # Load environment variables from .env file
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        # reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    return llm
