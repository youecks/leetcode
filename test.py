def longestCommonPrefix(arr):
    pref = ''
    for i in range(len(arr[1])):
        for el in arr:
           if el[i] != arr[1][i]:
               return pref
        pref += el[i]


print(longestCommonPrefix(["flower","flow","floght"]))