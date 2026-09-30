class Candidate:
    def __init__(self, name, age):
        self.name = name
        self.age = age

candidate = Candidate("Alice", 25)
print(f"Name: {candidate.name}")
print(f"Age: {candidate.age}")
