from fastapi import FastAPI
import torch
import tensorflow as tf
import pickle
from pydantic import BaseModel
from fastapi.responses import FileResponse
from transformers import T5ForConditionalGeneration, TFT5ForConditionalGeneration

from models.t5 import infer_t5

import os

app = FastAPI()


class Input(BaseModel):
    model: str
    language: str
    text: str


@app.head("/ping")
@app.post("/invocations")
def invoke(inp: Input):
    input_dict = inp.dict()
    print(input_dict["language"])
    text = input_dict["language"] + " : " + input_dict["text"]

    # Load model
    if 'torch' in input_dict['model']:
        if 'quantized' in input_dict['model']:
            model = T5ForConditionalGeneration()
            state_dict = torch.load(input_dict['model'])
            model.load_state_dict(state_dict=state_dict)
        else:
            model = T5ForConditionalGeneration.from_pretrained(input_dict["model"])
        framework = 'torch'
    else:
        if 'quantized' in input_dict['model']:
            with open(input_dict['model'], "rb") as f:
                state_dict = pickle.load(f)
            model = tf.lite.Interpreter(model_content=state_dict)
            model.allocate_tensors()
        else:
            model = TFT5ForConditionalGeneration.from_pretrained(input_dict["model"])
        framework = 'tf'

    # Translate text
    output = infer_t5(model, framework, text)
    print(' translated_text : ', output)
    return output


@app.get("/ping")
def ping():
    return 'ping'


header = "timestamp,project_name,run_id,duration,emissions,emissions_rate,cpu_power,gpu_power,ram_power," \
         "cpu_energy,gpu_energy,ram_energy,energy_consumed,country_name,country_iso_code,region,cloud_provider," \
         "cloud_region,os,python_version,cpu_count,cpu_model,gpu_count,gpu_model,longitude,latitude," \
         "ram_total_size,tracking_mode,on_cloud"
example = "2022-11-26T10:32:27,codecarbon,cc2e23fa-52a8-4ea3-a4dc-f039451bcdc4,0.871192216873169," \
          "4.1067831054495705e-07,0.0004713980480897,7.5,0.0,1.436851501464844,1.8141875664393104e-06," \
          "0,3.472772259025685e-07,2.161464792341879e-06,Spain,ESP,catalonia,,," \
          "Linux-5.15.0-53-generic-x86_64-with-glibc2.35,3.10.6,4,AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx," \
          ",,2.2586,41.9272,3.83160400390625,machine,N"


@app.get("/results", responses={200: {"description": "CSV file containing all of the information collected from "
                                                     "each inference call made until now.",
                                      "content": {"text/csv": {"example": header + "\n" + example}}
                                      }
                                }
         )
def results():
    file_path = "emissions.csv"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/csv")
    return {"error": "File not found!"}
