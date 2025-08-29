A = [[1, 2, 3],
     [4, 5, 6]]

B = [[7, 8, 9],
     [10, 11, 12]]

C = [[1, 2],
     [3, 4],
     [5, 6]]

result_add = []
for i in range(len(A)):
    row = []
    for j in range(len(A[0])):
        row.append(A[i][j] + B[i][j])
    result_add.append(row)

result_mul = []
for i in range(len(A)):
    row = []
    for j in range(len(C[0])):
        sum_product = 0
        for k in range(len(A[0])):
            sum_product += A[i][k] * C[k][j]
        row.append(sum_product)
    result_mul.append(row)

print("Addition of A and B:")
print(result_add)

print("Multiplication of A and C:")
print(result_mul)
