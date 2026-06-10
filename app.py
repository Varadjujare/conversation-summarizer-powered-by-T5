# fast api related code

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates # for UI part
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initilize fast api app
app = FastAPI(title="Hugging Face Text Summarizer API", description="""
                Text Summarizer 
                API built with FastAPI and Hugging Face Transformers
                """)

# Load the model 
model = T5ForConditionalGeneration.from_pretrained('./saved_summary_model')
tokenizer = T5Tokenizer.from_pretrained('t5-small')  # ← use t5-small, not saved model


# Fine Tuning the model

import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = model.to(device)

# Templating
templates = Jinja2Templates(directory=".")


# input schema
class dialougeInput(BaseModel):
    dialogue: str

def clean_data(text):
  text = re.sub(r"\r\n", " ", text) # lines removed
  text = re.sub(r"\s+", " ", text) # spaces removed (corrected regex)
  text = re.sub(r"<.*?>", " ", text) # html tags are removed
  text = text.strip().lower()
  return text



def summarize_dialogue(dialogue : str)->str:
    dialogue = clean_data(dialogue)  # clean

    # Tokenize
    inputs = tokenizer(
        dialogue,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)  # move inputs to device here

    # Generate summary token ids
    model.to(device)
    target = model.generate(
      input_ids=inputs['input_ids'],
      attention_mask=inputs['attention_mask'],
      max_length=200,       # increase
      min_length=60,        # force longer summary
      num_beams=4,
      length_penalty=2.0,   # increase to encourage longer output
      early_stopping=True
  )

    # Decode token ids to text
    summary = tokenizer.decode(target[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return summary


# API ENDPOINTS

@app.post("/summarize/")
async def summarize(dialogue: dialougeInput):
    summary = summarize_dialogue(dialogue.dialogue)  # fixed
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")