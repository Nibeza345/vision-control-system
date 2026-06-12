import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('../data/db/face_db.pkl', 'rb') as f:
    db = pickle.load(f)

genuine_scores = []
impostor_scores = []

# Genuine pairs
for name, embs in db.items():
    embs = np.array(embs)
    if len(embs) >= 2:
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                genuine_scores.append(np.dot(embs[i], embs[j]))

# Impostor pairs (limited for speed)
names = list(db.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        e1 = np.array(db[names[i]])
        e2 = np.array(db[names[j]])
        for a in e1:
            for b in e2[:8]:
                impostor_scores.append(np.dot(a, b))

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

print(f"Genuine mean: {genuine_scores.mean():.3f} ± {genuine_scores.std():.3f}")
print(f"Impostor mean: {impostor_scores.mean():.3f} ± {impostor_scores.std():.3f}")

plt.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine', color='blue')
plt.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor', color='orange')
plt.axvline(0.62, color='red', linestyle='--', label='Suggested Threshold 0.62')
plt.legend()
plt.xlabel('Cosine Similarity')
plt.title('Threshold Evaluation')
plt.show()