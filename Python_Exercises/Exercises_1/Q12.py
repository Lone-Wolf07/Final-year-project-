a = int(input("Enter numerator: "))
b = int(input("Enter denominator: "))

try:
    result = a / b
    print(f"Result: {result}")
except ZeroDivisionError:
    print("Error: Cannot divide by zero")
