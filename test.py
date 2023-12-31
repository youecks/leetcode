def twoSum(nums, target):
    hashmap = {}
    
    for idx, num in enumerate(nums):
        compl = target - num
        if compl in hashmap:
            return (hashmap[compl], idx)
        
        hashmap[num] = idx
