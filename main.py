from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


os.environ['HF_TOKEN'] = ""
HF_TOKEN = os.environ["HF_TOKEN"]

model_id = "microsoft/Phi-3.5-mini-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_folder="offload_folder"
)


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]

def clean_response(text: str) -> str:
    clean_text = text.replace("Assistant:", "").replace("Chatbot:", "")
    clean_text = clean_text.split("[SYS]")[0].split("[DATA]")[0].strip()
    return " ".join(clean_text.splitlines()).strip()

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    input_text = request.messages[-1].content
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            max_new_tokens=256
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    clean_output_text = clean_response(output_text)

    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": clean_output_text
            }
        }]
    }
@app.get("/")
def root():
    return {"message": "Server is running!"}

if __name__ == "__main__":
    import uvicorn
    torch.cuda.empty_cache()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)