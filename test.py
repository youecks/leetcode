###################
## Valid Anagram ##
###################

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        # initialize hashmap
        hmap_s ={}
        hmap_t = {}

        # add string characters to hashmap
        for i in range(len(s)):
            hmap_s[s[i]] = 1 + hmap_s.get(s[i], 0)
            hmap_t[t[i]] = 1 + hmap_t.get(t[i], 0)

        if hmap_s == hmap_t:
            return True

