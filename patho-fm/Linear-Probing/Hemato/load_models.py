import os
import torch
import timm
from transformers import CLIPModel
from transformers import AutoModel
import torch.nn as nn
import huggingface_hub
print(huggingface_hub.__version__)
from huggingface_hub import login
torch.cuda.empty_cache()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clone Prov-GigaPath repository
prov_gigapath_dir = "models/prov_gigapath"
if not os.path.exists(prov_gigapath_dir):
    os.system("git clone https://github.com/prov-gigapath/prov-gigapath models/prov_gigapath")
else:
    print(f"Directory {prov_gigapath_dir} already exists, skipping git clone.")

# Login to Hugging Face
login("hf_token")  #Your token

model_gigapath = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
model_gigapath.to(device)

# Load UNI2-h model
uni_dir = "models/UNI"
if not os.path.exists(uni_dir):
    os.system("git clone https://github.com/mahmoodlab/UNI.git models/UNI")
else: 
    print(f"Directory {uni_dir} already exists, skipping git clone.")

timm_kwargs = {
    'img_size': 224,
    'patch_size': 14,
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5,
    'embed_dim': 1536,
    'mlp_ratio': 2.66667*2,
    'num_classes': 0,
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked,
    'act_layer': torch.nn.SiLU,
    'reg_tokens': 8,
    'dynamic_img_size': True
}
model_uni = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
model_uni.to(device)

# Load CONCH model
conch_dir = "models/CONCH"
if not os.path.exists(conch_dir):
    os.system("git clone https://github.com/mahmoodlab/CONCH.git models/CONCH")
    os.system("pip install git+https://github.com/Mahmoodlab/CONCH.git")
else:
    print(f"Directory {conch_dir} already exists, skipping git clone.")

from conch.open_clip_custom import create_model_from_pretrained

# show all jupyter output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

model_cfg = 'conch_ViT-B-16'

model_conch, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
model_conch = model_conch.to(device)

# Load PLIP model
model_plip = CLIPModel.from_pretrained("vinid/plip")
model_plip.to(device)

# Load PhikonV2 model
model_phikon = AutoModel.from_pretrained("owkin/phikon-v2")
model_phikon.to(device)


# Load Hibou-L model
hibou_dir = "models/Hibou"
if not os.path.exists(hibou_dir):
    os.system("pip install lfs")
    os.system("git clone https://huggingface.co/histai/cellvit-hibou-L models/Hibou")
else:
    print(f"Directory {hibou_dir} already exists, skipping git clone.")

model_hibou = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True) #, force_download=True)
model_hibou.to(device)

# Load DinoBloom model
try:
    model_dinobloom = timm.create_model(
        model_name="hf-hub:1aurent/vit_base_patch14_224.dinobloom",
        pretrained=True,
        img_size = 224
    ).eval()
    print("DinoBloom model loaded successfully.")
except Exception as e:
    print(f"Failed to load DinoBloom model: {e}")
    model_dinobloom = None
model_dinobloom.to(device)

# Load RedDino model from Hugging Face Hub
model_reddino = timm.create_model("hf_hub:Snarcy/RedDino-base", pretrained=True)
model_reddino.eval()
model_reddino.to(device)


#Load PathOrchestra model
model_pathorchestra = timm.create_model(
    "hf-hub:AI4Pathology/PathOrchestra",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True,
)

model_pathorchestra.eval()
model_pathorchestra.to(device)
