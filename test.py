def longestCommonPrefix(arr):
    pref = ''

    for idx in range(len(arr[0])):
        for el in arr:
        #      num         num
            if idx == len(el):
                return pref
            if el[idx] != arr[0][idx]:
                return pref
        pref += el[idx]



print(longestCommonPrefix(["dog","dacecar","dar"]))