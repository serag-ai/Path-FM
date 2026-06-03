import os
import torch
import timm
from transformers import CLIPModel
from transformers import AutoModel
import histoencoder.functional as F
import torch.nn as nn
from pathlib import Path
import huggingface_hub
from huggingface_hub import login
torch.cuda.empty_cache()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Load HistoEncoder

model_prostat_small = F.create_encoder(model_name="prostate_small")
model_prostat_medium = F.create_encoder(model_name="prostate_medium")
model_prostat_small.to(device)
model_prostat_medium.to(device)