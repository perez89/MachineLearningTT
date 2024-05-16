# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

data = "hello"
# string assign
print("testing anaconda " + data +  " ")

print(data[0] + "sdf ")
print(len(data))

value = 123.1

print(value)

value = 10
print(value)

#boolean assignment
a = True
b = False
print(a, b)

d,e,f, = 1,2,3
print(d,e,f)

#non assign
g = None
print(g)

value = 301
if value == 99:
    print('That is fast')
elif value > 200:
    print("That is too fast")
else:
    print('This is safe')
    
for i in range(100):
    print(i)
    
a = (1,2,3)
print(a)

mylist = [1,2,3]
print("zeroth value: %d" % mylist[0])

mylist.append(4)
print('List length: %d' % len(mylist))

for value in mylist:
    print (value)
    
mydict = {'a': 1, 'b': 2 , 'c': 3}
print("A value %d" % mydict['a']) 

mydict['a'] = 5

print("A value %d" % mydict['a']) 
print("keys: %s" % mydict.keys())
 
print("keys: %s" % mydict.values())

for v in mydict.values():
    print(v)
    
for k in mydict.keys():
    print(mydict[k])
    
#functions
def mysum(x,y):
    return x+y;

result = mysum(4,5)
print(result)
