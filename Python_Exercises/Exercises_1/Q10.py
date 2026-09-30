filename = input("Enter the filename: ")

with open(filename, "r") as file:
    contents = file.read()

print(contents)
