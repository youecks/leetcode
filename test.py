from collections import defaultdict

def groupAnagrams(strs):
    my_dict = defaultdict(list)
    # for each string in the array
    for string in strs:

        # array of 26 0's
        count = [0]*26
        
        # for each character in the string
        for char in string:
            
            # asky value for the char is an integer 
            # count[0] += 1
            # add 1 to the index listed
            count[ord(char) - ord('a')] += 1
            
            # key is the array    value is the string
        my_dict[tuple(count)].append(string)

    # return all values in the dictionary
    return my_dict.values()

print(groupAnagrams(["eat","tea","tan","ate","nat","bat", "tab", "by"]))