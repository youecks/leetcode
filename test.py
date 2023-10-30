#############
## Two Sum ##
#############

def twoSum(nums, target):
    prev_map = {}

    for i, num in enumerate(nums, 0):
        pair = target - num
        print(pair)

        if pair in prev_map:
            return (prev_map[pair], i)
       
        prev_map[num] = i

# n = number of elements 
# Time: O(n)
# Space: O(n)