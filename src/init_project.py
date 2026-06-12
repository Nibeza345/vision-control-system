import os

os.makedirs("../data/enroll", exist_ok=True)
os.makedirs("../data/db", exist_ok=True)
os.makedirs("../data/logs", exist_ok=True)
os.makedirs("../models", exist_ok=True)
print("Folders created: data/enroll, data/db, data/logs, models")