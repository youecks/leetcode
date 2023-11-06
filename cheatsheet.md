1. start with accessing then build out the syntax from there
2. Comment each line of code
3. Consider what each piece of code evaluates to
4. give names to things that make sense

##########################################################
##                      Array/List                      ##
##########################################################

Array - used to store collections of data, each index is an item

fruits = ['apple', 'banana', 'cherry']

Traverse/loop
    for item in array:  #loops through arr
    for item in array[0]: #loops through 1st item (letter in str)

Access
    fruits[0]       #access first item
    fruits[0][1]    #access letter of first item

Add
    fruits.append('orange')
    fruits.insert(1, 'orange')

Remove
    fruits.remove('banana')
    fruits.pop(1)
    del fruits[0]



##########################################################
##                      String                          ##
##########################################################


Traverse/loop
    for char in str:
Access
Add
Remove