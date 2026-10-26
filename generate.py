"""Refactored from the Basic Text Generation examples in the included PDF."""
import argparse

def generate_prompts(model_name, prompts, max_new_tokens=80, temperature=0.7, seed=42):
    if not prompts or any(not p.strip() for p in prompts): raise ValueError('Prompts must not be empty')
    if not 1 <= max_new_tokens <= 512 or temperature <= 0: raise ValueError('Invalid generation parameters')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    set_seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    results=[]
    for prompt in prompts:
        inputs=tokenizer(prompt,return_tensors='pt')
        limit=getattr(model.config,'max_position_embeddings',2048)
        if inputs['input_ids'].shape[1]+max_new_tokens>limit:raise ValueError('Prompt exceeds model context')
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=max_new_tokens,
                temperature=temperature,do_sample=True,pad_token_id=tokenizer.pad_token_id)
        results.append(tokenizer.decode(output[0],skip_special_tokens=True))
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='EleutherAI/gpt-neo-125m')
    p.add_argument('--prompt',action='append')
    p.add_argument('--max-new-tokens',type=int,default=80)
    p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    prompts=a.prompt or ['List the top three uses of apple.',
        'List the top three uses of apple fruit.','List the top three uses of apple drink.']
    for prompt,result in zip(prompts,generate_prompts(a.model,prompts,a.max_new_tokens,a.temperature,a.seed)):
        print(f'\nPROMPT: {prompt}\nOUTPUT: {result}')
if __name__=='__main__':main()
