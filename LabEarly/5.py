students = {}

def add_student(name, marks):
    students[name] = marks

def update_marks(name, new_marks):
    if name in students:
        students[name] = new_marks
    else:
        print(f"Student {name} not found.")

def delete_student(name):
    if name in students:
        del students[name]
    else:
        print(f"Student {name} not found.")

def find_topper():
    if not students:
        print("No students found.")
        return
    topper = max(students, key=students.get)
    print(f"The topper is {topper} with {students[topper]} marks.")

#check kr rahay
add_student("Alice", 85)
add_student("Bob", 92)
add_student("Charlie", 78)
update_marks("Alice", 90)
delete_student("Charlie")
find_topper()
