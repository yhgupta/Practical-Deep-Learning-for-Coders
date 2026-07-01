from transformers import pipeline

generator = pipeline("text-generation",  model="bert-base-cased")
result = generator(
    "I lost interest in love, The person who wants love most, leaves the love at the end",
    max_length=50,
    num_return_sequences=1,
)

print(result)