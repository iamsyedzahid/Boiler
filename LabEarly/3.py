students = [
    ("Alice", 85),
    ("Bob", 92),
    ("Charlie", 78),
    ("David", 90),
    ("Eva", 88),
    ("Frank", 75)
]

for i in range(len(students)):
    for j in range(len(students) - 1):
        if students[j][1] < students[j + 1][1]:
            students[j], students[j + 1] = students[j + 1], students[j]

for i in range(3):
    print(students[i])