def removeElement(nums, val):
    k = 0
    for i in range(len(nums)):
        # 0, 1, 2, 3

        if nums[i] != val:
        #   2 not equal to 3

            nums[k] = nums[i]
            # 2 = 3

            k += 1
            # non val nums + 1
    print(k)
    return nums
    

print(removeElement([0,1,2,2,3,0,4,2], 2))
#                   