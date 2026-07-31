from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

text = (
    "TOO TRUE TO BE GOOD ACT I Night. One of the best bedrooms in one of the best suburban villas in one of the richest cities in England. "
    "A young lady with an unhealthy complexion is asleep in the bed. A small table at the head of the bed, convenient to her "
    "right hand, and crowded with a medicine bottle, a measuring glass, a pill box, a clinical thermometer in a glass of water, "
    "a half read book with the place marked by a handkerchief, a powder puff and handmirror, and an electric bell handle on a flex, "
    "shews that the bed is a sick bed and the young lady an invalid."
)
model_id = 'facebook/nllb-200-distilled-600M'

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_id)
print('Loading model...')
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
print('Tokenizer type:', type(tokenizer).__name__)
source_code = 'eng_Latn'
target_code = 'kin_Latn'
print('Source code:', source_code)
print('Target code:', target_code)
print('Target token id:', tokenizer.convert_tokens_to_ids(target_code))
print('Target token string:', tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(target_code)))
print('Bos token id:', tokenizer.bos_token_id)
print('EOS token id:', tokenizer.eos_token_id)
print('Pad token id:', tokenizer.pad_token_id)
print('UNK token id:', tokenizer.unk_token_id)

if callable(getattr(tokenizer, 'set_src_lang_special_tokens', None)):
    tokenizer.set_src_lang_special_tokens(source_code)
if callable(getattr(tokenizer, 'set_tgt_lang_special_tokens', None)):
    tokenizer.set_tgt_lang_special_tokens(target_code)

inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
print('Input ids shape:', inputs['input_ids'].shape)
print('Input sample decoded:', tokenizer.decode(inputs['input_ids'][0][:80], skip_special_tokens=False))
forced_bos = tokenizer.convert_tokens_to_ids(target_code)
print('Forced bos id:', forced_bos)
print('Forced bos token:', tokenizer.convert_ids_to_tokens(forced_bos))

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos,
        max_new_tokens=512,
        num_beams=3,
        early_stopping=True,
    )

print('Generated shape:', generated.shape)
print('Generated ids:', generated[0].tolist())
print('Decoded output skip special tokens:', tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
print('Decoded output keep special tokens:', tokenizer.batch_decode(generated, skip_special_tokens=False)[0])
print('Output token count:', len(tokenizer.batch_decode(generated, skip_special_tokens=False)[0].split()))
