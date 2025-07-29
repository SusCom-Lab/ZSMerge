from mergekv import AttentionForward


batch_prompts = [
    "What is the capital of France?",
    # "Explain the theory of relativity.",
    # "Describe the process of photosynthesis.",
    # "Who is the author of Pride and Prejudice?",
]
output_len = 6

tokenizer, model = AttentionForward.model_load(model_name="Qwen/Qwen1.5-7B", merge=False)
model.eval().to(AttentionForward.device)

tokenizer.pad_token = tokenizer.eos_token
tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to(AttentionForward.device)
batch_input_ids = tokenized_prompts.input_ids

context_length = batch_input_ids.shape[-1]
output_max_len = context_length + output_len

output = model.generate(
    **tokenized_prompts,
    output_attentions = False,
    max_new_tokens=output_max_len,
    num_beams=1,
    do_sample=False,
    top_p=None,
    temperature=1.0,
    min_length=context_length+1,
    eos_token_id=[tokenizer.eos_token_id]
)

batch_outputs =tokenizer.batch_decode(output[:, context_length:], skip_special_tokens=True)
for q, a in zip(batch_prompts, batch_outputs):
    print("-" * 20, f"Q:\n\t{q}\nA:\n\t{a}", sep="\n")
