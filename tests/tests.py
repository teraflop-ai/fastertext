from fastertext import load_model

model = load_model("model_v3.bin")
labels, probs = model.batch(["hello world", "bonjour le monde"], k=2)

id2label = model.get_labels()
print(id2label[int(labels[0, 0])], probs[0, 0])