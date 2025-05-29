from mergekv import AttentionForward as AF

# 在现有代码中添加以下清理逻辑
import gc
import torch
import weakref
import functools

def cleanup_after_test(cls):
    """装饰器：确保测试类在测试完成后清理资源"""
    original_tearDownClass = getattr(cls, 'tearDownClass', None)
    
    @classmethod
    def tearDownClass(cls):
        cls_cleanup(cls)
        if original_tearDownClass:
            original_tearDownClass()
    
    cls.tearDownClass = tearDownClass
    return cls

def cls_init(cls):
    model_name = cls.model_name
    print(f"loading model: {model_name}")
    cls.device = AF.device
    tokenizer, model = AF.model_load(model_name=model_name, merge=False)
    model.eval().to(cls.device)
    # 使用weakref代理
    cls._model_ref = weakref.ref(model)
    cls._tokenizer_ref = weakref.ref(tokenizer)
    cls.model = cls._model_ref()
    cls.tokenizer = cls._tokenizer_ref()

# 在test_utils.py中添加通用清理方法
def cls_cleanup(cls):
    if hasattr(cls, '_model_ref'):
        model = cls._model_ref()
        if model is not None:
            model.to('cpu')
            del model
            cls.model = None
    if hasattr(cls, '_tokenizer_ref'):
        tokenizer = cls._tokenizer_ref()
        if tokenizer is not None:
            del tokenizer
            cls.tokenizer = None
    torch.cuda.empty_cache()
    gc.collect()


def gen_equal(self):
    batch_prompts = [
        "What is the capital of France?",
        # "Explain the theory of relativity.",
        # "Describe the process of photosynthesis.",
        "Who is the author of Pride and Prejudice?",
    ]
    output_len = 16

    tokenizer, model = self.tokenizer, self.model
    model.eval()

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to(self.device)
    context_length = tokenized_prompts.input_ids.shape[-1]

    AF.change_mode(merge=False)
    output = model.generate(
        **tokenized_prompts.copy(),
        output_attentions = False,
        max_new_tokens=output_len,
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
    
    AF.change_mode(merge=True, cache_budget=100)
    # with self.assertRaises(NotImplementedError):
    #     output = model.generate(
    #         **tokenized_prompts,
    #         output_attentions = False,
    #         max_new_tokens=output_len,
    #         num_beams=1,
    #         do_sample=False,
    #         top_p=None,
    #         temperature=1.0,
    #         min_length=context_length+1,
    #         eos_token_id=[tokenizer.eos_token_id]
    #     )

    for idx, (q, a) in enumerate(zip(batch_prompts, batch_outputs)):
        tokenized_prompts = tokenizer([q], padding="longest", return_tensors="pt", add_special_tokens=True).to(self.device)
        context_length = tokenized_prompts.input_ids.shape[-1]

        AF.change_mode(merge=True, cache_budget=100)
        output = model.generate(
            **tokenized_prompts,
            output_attentions = False,
            max_new_tokens=output_len,
            num_beams=1,
            do_sample=False,
            top_p=None,
            temperature=1.0,
            min_length=context_length+1,
            eos_token_id=[tokenizer.eos_token_id]
        )
        batch_output_ =tokenizer.batch_decode(output[:, context_length:], skip_special_tokens=True)
        print("+" * 20, f"Q:\n\t{q}\nA:\n\t{batch_output_[0]}", sep="\n")
        self.assertEqual(batch_output_[0], a)

        AF.change_mode(merge=True, cache_budget=10)
        output = model.generate(
            **tokenized_prompts,
            output_attentions = False,
            max_new_tokens=output_len,
            num_beams=1,
            do_sample=False,
            top_p=None,
            temperature=1.0,
            min_length=context_length+1,
            eos_token_id=[tokenizer.eos_token_id]
        )
        batch_output_ =tokenizer.batch_decode(output[:, context_length:], skip_special_tokens=True)
        print("~" * 20, f"Q:\n\t{q}\nA:\n\t{batch_output_[0]}", sep="\n")

