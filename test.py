def longestCommonPrefix(arr):
    pref = ""

    
    for i in range(len(arr[0])):
    
        for idx in arr:
        
            if idx[i] == arr[0][i] and idx[i] != len(idx):
                print(len(idx))
            




longestCommonPrefix(["flower","flow","flight"])