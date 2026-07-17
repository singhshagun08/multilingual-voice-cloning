import pandas as pd

files = {
    "English (CPU)": "benchmark_english_cpu.csv",
    "Arabic (CPU)": "benchmark_arabic.csv",
    "Hindi (CPU)": "benchmark_hindi.csv",
}

rows = []
for label, path in files.items():
    df = pd.read_csv(path)
    row = df.iloc[0]
    rows.append({
        "Language": label,
        "Latency (s)": round(row["Latency"], 2),
        "Audio Duration (s)": round(row["Audio Duration"], 2),
        "RTF": round(row["RTF"], 2),
        "WER": round(row["WER"], 3),
        "CER": round(row["CER"], 3),
        "Similarity": round(row["Similarity"], 3),
        "Meets RTF<0.5?": "No",
        "Meets Similarity>0.75?": "No",
    })

result = pd.DataFrame(rows)
print(result.to_string(index=False))
result.to_csv("full_comparison_cpu.csv", index=False)
print("\nSaved full_comparison_cpu.csv")

print("\n--- GPU reference point (Colab T4, English only) ---")
print("Latency: 6.6s | RTF: 1.17 | WER: 0.0 | CER: 0.0 | Similarity: 0.488")
print("(Not directly comparable to CPU rows above -- shown separately for hardware context)")
