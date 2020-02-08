"""
def classify(num):
    
    #determine if the number is divisible by 2, 3 or 5
    
    if (num % 2 == 0):
        print('2')
    if (num % 3 == 0):
        print('3')
    if (num % 5 == 0):
        print('5')

for i in range(5):
    print(i)
print(range(5))

[1, 2, 3] - 1
"""
#This file was used to help tutor students in python.
n = 50
count = 0
for i in range(0, n):
    for j in range(0, i):
        count+=1

print(n*(n-1)/2)
print(count)