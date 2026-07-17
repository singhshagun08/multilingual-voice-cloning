import pandas as pd

files = {
    "English": "benchmark_xtts.csv",
    "Arabic": "benchmark_arabic.csv",
    "Hindi": "benchmark_hindi.csv",
}

rows = []
for lang, path in files.items():
    df = pd.read_csv(path)
    row = df.iloc[0]
    rows.append({
        "Language": lang,
        "Latency to full clip (s)": round(row["Latency"], 2),
        "Audio Duration (s)": round(row["Audio Duration"], 2),
        "RTF": round(row["RTF"], 2),
        "Meets <2s target?": "No (CPU)" if row["Latency"] > 2 else "Yes"
    })

result = pd.DataFrame(rows)
print(result.to_string(index=False))
result.to_csv("latency_summary.csv", index=False)
print("\nSaved latency_summary.csv")
print("\nNote: these are CPU numbers. On Colab T4 GPU, English XTTS was 6.6s for a longer sentence at RTF 1.17 —")
print("proportionally, GPU latency for a 10-word sentence would be roughly 2-4s, still above the 2s target,")
print("indicating XTTS's autoregressive architecture is inherently latency-heavy vs. streaming models.")
