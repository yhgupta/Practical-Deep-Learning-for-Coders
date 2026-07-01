prompt = "I want to become extremely successful person."

from transformers import pipeline

generator = pipeline("text-generation",  model="LLM Course/my_awesome_eli5_clm-model")
result = generator(prompt)

print(result)



from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("LLM Course/my_awesome_eli5_clm-model")
inputs = tokenizer(prompt, return_tensors="pt").input_ids

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("LLM Course/my_awesome_eli5_clm-model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

result_2 = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(outputs)
print(result_2)