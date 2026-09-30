import json
import os
from cryptography.fernet import Fernet

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(OUTPUT_DIR, "secret.key")
DATA_FILE = os.path.join(OUTPUT_DIR, "passwords.json")

def load_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as file:
            file.write(key)
    with open(KEY_FILE, "rb") as file:
        return file.read()

def load_data():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, "r") as file:
        return json.load(file)

def save_data(data):
    with open(DATA_FILE, "w") as file:
        json.dump(data, file)

def add_password(fernet, site, password):
    data = load_data()
    data[site] = fernet.encrypt(password.encode()).decode()
    save_data(data)

def get_password(fernet, site):
    data = load_data()
    if site not in data:
        return None
    return fernet.decrypt(data[site].encode()).decode()

fernet = Fernet(load_key())

choice = input("Add or Get password? (add/get): ")
site = input("Enter site name: ")

if choice == "add":
    password = input("Enter password: ")
    add_password(fernet, site, password)
    print("Password saved")
else:
    password = get_password(fernet, site)
    print("Password:", password if password else "Not found")
