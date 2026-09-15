# bench_models.py
import fasttext
import pyperf
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

if __name__ == "__main__":
    runner = pyperf.Runner()
    args = runner.parse_args()
    runner.metadata.update(model=MODEL, k=K, texts=len(texts))

    ft = fasttext.load_model(MODEL)
    rt = load_model(MODEL)

    b_ft = runner.bench_func("fasttext", lambda: ft.predict(texts, k=K))
    b_rt = runner.bench_func("fastertext", lambda: rt.batch(texts, k=K))

    if not args.worker:
        t_ft, t_rt = b_ft.mean(), b_rt.mean()
        n = len(texts)

        print(f"fasttext:   {t_ft:.3f}s  {n / t_ft:,.0f} texts/s")
        print(f"fastertext: {t_rt:.3f}s  {n / t_rt:,.0f} texts/s")
        print(f"speedup: {t_ft / t_rt:.1f}x")

        sample = texts[:1000]
        ft_labels, _ = ft.predict(sample, k=1)
        labels, _ = rt.batch(sample, k=1)
        id2lab = rt.get_labels()
        agree = sum(
            id2lab[int(a)] == b[0]
            for a, b in zip(labels[:, 0], ft_labels)
        )
        print(f"top-1 agreement: {agree / len(sample):.1%}")