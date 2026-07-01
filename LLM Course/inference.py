prompt = "I am an astronaut."

from transformers import pipeline

generator = pipeline("text-generation",  model="./my_awesome_eli5_clm-model/checkpoint-3912")
result = generator(prompt)

print(result)



from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./my_awesome_eli5_clm-model/checkpoint-3912")
inputs = tokenizer(prompt, return_tensors="pt").input_ids

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./my_awesome_eli5_clm-model/checkpoint-3912")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

result_2 = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(outputs)
print(result_2)