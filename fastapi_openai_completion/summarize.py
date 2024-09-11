from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
client = OpenAI() # just export OPENAI_API_KEY into terminal to work

system_message = "You are a concise summarizer. Break down the following text into its main ideas, then output these main ideas as bullet points."

class SummaryRequest(BaseModel):
    text: str

class SummaryResponse(BaseModel):
    output: str


@app.post("/text_upload", response_model=SummaryResponse)
def summarize(request: SummaryRequest):
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system",
            "content": system_message},
            {"role":"user",
            "content":request.text}
        ]
    )

    summary = chat_completion.choices[0].message.content
    return SummaryResponse(output=summary)
