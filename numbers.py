# input = N, output = round

i = int(input('Enter a number'))

n = 0
z = 0
if (i > 28):
    diff = i - 28
    num = 0
    for j in range(29, 2000):

        j1 = j
        z = 0
        while(j1 > 0):
            z = z + int(j1 % 10)
            j1 = j1 / 10
        if (z <= 10):
            num = num + 1
        if num == diff:
            break
    i = j

i1 = i
while(i1 > 0):
    n = n + int(i1 % 10)
    i1 = i1 / 10


i = i * 10

ans = i + (10-n)

print(ans)
