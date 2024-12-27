from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-pt", device=0)

text = """
A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world. Konstantin Batygin did not set out to solve one of the solar system’s most puzzling mysteries when he went for a run up a hill in Nice.
"""

res = pipe(text,src_lang="en", tgt_lang="pt", max_length=400)
print(res) 