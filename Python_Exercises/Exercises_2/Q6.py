import re

pattern = r"^[\w.-]+@[\w.-]+\.\w+$"

email = input("Enter an email address: ")

if re.match(pattern, email):
    print("Valid email address")
else:
    print("Invalid email address")
