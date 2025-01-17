import torch
import sys
from transformers import BertTokenizerFast, BertForMaskedLM
import json

# Load pre-trained model tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Input sentence and mask
sentence = sys.argv[1]
mask = sys.argv[2]
sentence = sentence.replace(mask, '[MASK]')

# Tokenize the sentence
inputs = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
tokens_tensor = inputs['input_ids']
segments_tensors = inputs['token_type_ids']

# Load pre-trained model
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, token_type_ids=segments_tensors)

# Get the index of the masked token
masked_index = (tokens_tensor[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

# Get top predictions
num_of_res = 10
logits = predictions.logits[0, masked_index]
probs = torch.nn.functional.softmax(logits, dim=0)

top_indices = torch.argsort(probs, descending=True)[:num_of_res]
top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
top_probs = probs[top_indices]

# Combine tokens with their probabilities
with_prob = list(zip(top_tokens, top_probs.tolist()))
print(json.dumps(with_prob))
