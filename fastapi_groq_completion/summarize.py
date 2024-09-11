from fastapi import FastAPI
from groq import Groq
from pydantic import BaseModel
import os

app = FastAPI()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY") # export GROQ_API_KEY = "your api key" <in the terminal>
)

class SummaryRequest(BaseModel):
    text: str

class SummaryResponse(BaseModel):
    output: str

system_message = "You are a concise summarizer. Break down the following text into its main ideas, then output these main ideas as bullet points."



@app.post("/text_upload", response_model=SummaryResponse)
def summarize(request: SummaryRequest):
    chat_completion = client.chat.completions.create(
        messages = [
            {"role":"system",
            "content":system_message},
            {"role":"user",
             "content":request.text}
        ],
        model="llama3-8b-8192",
        temperature = 0,
        max_tokens=8192,
        top_p=1,
        stream=False
    )

    summary = chat_completion.choices[0].message.content
    return SummaryResponse(output=summary)