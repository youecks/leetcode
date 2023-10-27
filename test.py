########################
## Contains Duplicate ##
########################

def containsDuplicate(nums):
    hashset = set()
    
    for n in nums:
        if n in hashset:
            return True
        hashset.add(n)
    return False

containsDuplicate([1,2,3,1])

# n = input array
# Time: O(n)
# Space: O(1)