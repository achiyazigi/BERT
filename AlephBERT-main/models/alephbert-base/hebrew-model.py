import json
from transformers import BertForMaskedLM, BertTokenizerFast
import torch
import pickle
import io
import sys

torch.device('cpu')


# Custom Unpickler to load a model on CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


# Inputs
text = sys.argv[1]
mask = sys.argv[2]
original = sys.argv[3]

# Load tokenizer
alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')

# Load model (pretrained or custom)
if original == "true":
    alephbert = BertForMaskedLM.from_pretrained('onlplab/alephbert-base')
else:
    with open('../../AlephBERT-main/models/myFirstTune/tune_6_10.pkl', 'rb') as model_file:
        alephbert = CPU_Unpickler(model_file).load()
        alephbert.bert.encoder.gradient_checkpointing = False

alephbert.eval()

# Prepare the text with [MASK] token
hebrew_text = text.replace(mask, '[MASK]')
inputs = alephbert_tokenizer(hebrew_text, return_tensors='pt', add_special_tokens=True)

tokens_tensor = inputs['input_ids']
segments_tensors = inputs['token_type_ids']

# Predict masked tokens
with torch.no_grad():
    predictions = alephbert(tokens_tensor, token_type_ids=segments_tensors)

# Locate the [MASK] token index
masked_index = (tokens_tensor[0] == alephbert_tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

# Get the top predictions and probabilities
num_of_res = 10
logits = predictions.logits[0, masked_index]
probs = torch.nn.functional.softmax(logits, dim=0)

top_indices = torch.argsort(probs, descending=True)[:num_of_res]
top_tokens = alephbert_tokenizer.convert_ids_to_tokens(top_indices.tolist())
top_probs = probs[top_indices]

# Combine tokens with their probabilities
with_prob = list(zip(top_tokens, top_probs.tolist()))

# Output as JSON
print(json.dumps(with_prob))
