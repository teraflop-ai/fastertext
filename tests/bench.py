import time
import fasttext
from fastertext import load_model

MODEL = "model_v3.bin"
K = 10
texts = [
    "hello world how are you doing today my friend",
    "bonjour tout le monde comment allez vous aujourd'hui",
    "hola amigo como estas hoy que tal",
    "das ist ein wunderbarer tag heute nicht wahr",
    "il gatto dorme sul divano tutto il giorno",
] * 4_000

ft = fasttext.load_model(MODEL)
rt = load_model(MODEL)

def bench(fn, n=10):
    fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)

t_ft = bench(lambda: ft.predict(texts, k=K))
t_rt = bench(lambda: rt.batch(texts, k=K))

n = len(texts)
print(f"fasttext: {t_ft:6.3f}s  {n / t_ft:>12,.0f} texts/s")
print(f"fastertext: {t_rt:6.3f}s  {n / t_rt:>12,.0f} texts/s")
print(f"speedup: {t_ft / t_rt:.1f}x")

ft_labels, _ = ft.predict(texts[:1000], k=1)
labels, _ = rt.batch(texts[:1000], k=1)
id2lab = rt.get_labels()
agree = sum(id2lab[int(a)] == b[0] for a, b in zip(labels[:, 0], ft_labels))
print(f"top-1 agreement: {agree / 1000:.1%}")