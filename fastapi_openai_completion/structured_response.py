from pydantic import BaseModel
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
client = OpenAI()

# the model should follow the structure defined in the class, but it can be good to define what is meant by each attribute/ reinforce the structure in the prompt.
system_message = """
You are an expert at structured data extraction. Take the following unstructured text and convert it into the given structure:
subject: The main noun associated with the text
keywords: Words that are especially important to the main message of the text
abstract: A concise, one sentence overview of what the text is about
"""

class SummaryRequest(BaseModel):
    input: str

class Summary(BaseModel):
    subject: str
    keywords: list[str]
    abstract: str

@app.post("/upload_text", response_model=Summary)
def summarize(request: SummaryRequest):
    chat_completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": request.input},
        ],
        response_format=Summary
    )

    return chat_completion.choices[0].message.parsed
        


       

