import sys
sys.path.append('./src/')
import torch
from sentencepiece import SentencePieceProcessor
from model import *
import torch.nn.functional as F
import argparse
from tokenizer import Tokenizer


tokenizer_path = './tokenizer.model'

checkpoint_path = './ckpt.pt'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def remove_unwanted_prefix_from_state_dict(state_dict, unwanted_prefix):
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict


def load_model(checkpoint_path, device, unwanted_prefix='_orig_mod.'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['model_args'] if isinstance(checkpoint['model_args'], ModelArgs) else ModelArgs(**checkpoint['model_args'])
    model = Transformer(config)
    if checkpoint.get('lora_finetune'):
        apply_lora(
            model, 
            targets=checkpoint['lora_targets'],
            rank=checkpoint['lora_rank'],
            dropout=checkpoint['lora_dropout'],
            alpha=checkpoint['lora_alpha']
        )
    print(f"Number of parameters: {sum([p.nelement() for p in model.parameters()])}")
    state_dict = checkpoint['model']
    state_dict = remove_unwanted_prefix_from_state_dict(state_dict=state_dict, unwanted_prefix=unwanted_prefix)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model, checkpoint


def generate_paragraph(
    model, 
    prompt,
    max_new_tokens=400,
    temperature=0.1,
    top_k=10,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2
):
    tokenized_prompt = tokenizer.encode(prompt, bos=True, eos=False)
    tokenized_prompt = (torch.tensor(tokenized_prompt, dtype=torch.long, device=device)[None, ...])

    paragraph = []
    context_tokens = tokenized_prompt
    for _ in range(max_new_tokens):
        context_tokens = context_tokens[:, -min(model.params.max_seq_len, context_tokens.size(1)):]
        output = model(context_tokens)
        logits = output[:, -1, :]
        logits = logits / temperature

        for token_id in set(paragraph):
            logits[:, token_id] /= repetition_penalty

        if no_repeat_ngram_size > 0:
            generated_ngrams = [tuple(paragraph[i:i+no_repeat_ngram_size]) for i in range(len(paragraph) - no_repeat_ngram_size + 1)]
            for ngram in generated_ngrams:
                ngram_token_ids = torch.tensor([ngram], dtype=torch.long, device=device)
                if context_tokens.size(1) >= ngram_token_ids.size(1) and (context_tokens[:, -ngram_token_ids.size(1):] == ngram_token_ids).all(dim=1).any():
                    for token_id in ngram:
                        logits[:, token_id] /= repetition_penalty



        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        #print("next::::::", next_)
        
        context_tokens = torch.cat((context_tokens, next_token), dim=1)
        paragraph.append(next_token.item())

        print(tokenizer.decode(next_token.item()), end=" ")
        if next_token.item() == tokenizer.eos_id() or tokenizer.decode(paragraph[-3:]) == 'The end.':
            break
    return context_tokens, paragraph, tokenizer.decode(paragraph)

parser = argparse.ArgumentParser(description="Generate a story")

parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for generating the story")
parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for controlling randomness")
parser.add_argument("--top_k", type=int, default=10, help="Number of top-k candidates to consider")
parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for repeated tokens")
parser.add_argument("--no_repeat_ngram_size", type=int, default=2, help="Size of n-grams that should not be repeated")


args = parser.parse_args()

tokenizer = Tokenizer(tokenizer_path)
instruct_model, ckpt = load_model(
    checkpoint_path=args.model_path,
    device=device,
)

generate_paragraph(
    model=instruct_model,
    prompt=args.prompt,
    max_new_tokens=400,
    temperature=args.temperature,
    top_k=args.top_k,
    repetition_penalty=args.repetition_penalty,
    no_repeat_ngram_size=args.no_repeat_ngram_size
)



