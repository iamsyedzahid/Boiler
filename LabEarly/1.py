password = input("Enter your password: ")


has_letter = False
has_number = False
has_special = False


for char in password:
    if char.isalpha():
        has_letter = True
    if char.isdigit():
        has_number = True
        has_special = True

if has_letter and has_number and has_special:
    print("Valid password")
else:
    print("Invalid password")

    if not has_letter:
        print("- Missing a letter")
    if not has_number:
        print("- Missing a number")
    if not has_special:
        print("- Missing a special character (@, #, $, %)")