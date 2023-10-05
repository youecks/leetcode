###########################
## Linked List Traversal ##
###########################

# Iterative
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

a = Node('A')
b = Node('B')
c = Node('C')
d = Node('D')

a.next = b
b.next = c
c.next = d

# Iterative Traversal
def print_list(head):
    current = head
    while current is not None:
        print(current.val)
        current = current.next

# Recursive Traversal
def print_list(head):
    if head is None:
        return 
    print(head.val)
    print_list(head.next)

print_list(b)


########################
## Linked List Values ##
########################

# Write a function, linked_list_values, that takes in the head of a linked list as an argument. The function should return a list containing all values of the nodes in the linked list.

# Iterative
def linked_list_values(head):
  values = []
  current = head
  while current is not None:
    values.append(current.val)
    current = current.next
  return values

# Recursive

def linked_list_values(head):
  values = []
  _linked_list_values(head, values)
  return values

def _linked_list_values(head, values):
  if head is None:
    return
  values.append(head.val)
  _linked_list_values(head.next, values)


##############
## Sum List ##
##############

# Iterative
class Node:
  def __init__(self, val):
    self.val = val
    self.next = None

def sum_list(head):
  
  current = head
  total_sum = 0
  
  while current is not None:
    total_sum += current.val
    current = current.next 
  
  
  return total_sum

# n = number of nodes
# Time: O(n)
# Space: O(1)


# Recursive
def sum_list(head):
  if head is None:
    return 0
  return head.val + sum_list(head.next)

# n = number of nodes
# Time: O(n)
# Space: O(n)


######################
## Linked List Find ##
######################

class Node:
  def __init__(self, val):
    self.val = val
    self.next = None

a = Node("a")
b = Node("b")
c = Node("c")
d = Node("d")

a.next = b
b.next = c
c.next = d

# Iterative
def linked_list_find(head, target):
  
    current = head
  
    while current is not None:
        if current.val == target:
            return True 
        current = current.next 
    return False

# n = number of nodes
# Time: O(n)
# Space: O(1)

# Recursive

def linked_list_find(head, target):
  if head is None:
    return False
  if head.val == target:
    return True
  return linked_list_find(head.next, target)

# n = number of nodes
# Time: O(n)
# Space: O(n)


####################
## Get Node Value ##
####################

# Recursive

class Node:
   def __init__(self, val):
      self.val = val
      self.next = None

def get_node_value(head, index):
  if head is None:
    return None
  if index == 0:
     return head.val
  
  return get_node_value(head.next, index - 1)

# Iterative

class Node:
  def __init__(self, val):
    self.val = val
    self.next = None
    
def get_node_value(head, index):
  current = head
  count = 0
  
  while current is not None:
    if count == index:
      return current.val
    count += 1
    current = current.next
  return None
    

##################
## Reverse List ##
##################

# Iterative
class Node:
  def __init__(self, val):
    self.val = val
    self.next = None
    
def reverse_list(head):
  prev = None 
  current = head
  while current is not None:
      next = current.next
      current.next = prev
      prev = current
      current = next
  return prev

# Recursive

def reverse_list(head, prev = None):
   if head is None:
      return prev
   next = head.next
   head.next = prev
   return reverse_list(next, head)


#################
## Zipper List ##
#################

# Iterative

def zipper_lists(head_1, head_2):
  tail = head_1
  current_1 = head_1.next
  current_2 = head_2
  count = 0
  while current_1 is not None and current_2 is not None:
    if count % 2 == 0:
      tail.next = current_2
      current_2 = current_2.next
    else:
      tail.next = current_1
      current_1 = current_1.next
    tail = tail.next
    count += 1
    
  if current_1 is not None:
    tail.next = current_1
  if current_2 is not None:
    tail.next = current_2
    
  return head_1

# n = length of list 1
# m = length of list 2
# Time: O(min(n, m))
# Space: O(1)

# Recursive

def zipper_lists(head_1, head_2):
  if head_1 is None and head_2 is None:
    return None
  if head_1 is None:
    return head_2
  if head_2 is None:
    return head_1
  next_1 = head_1.next
  next_2 = head_2.next
  head_1.next = head_2
  head_2.next = zipper_lists(next_1, next_2)
  return head_1  

# n = length of list 1
# m = length of list 2
# Time: O(min(n, m))
# Space: O(min(n, m))


#################
## Merge Lists ##
#################

# Iterative 

class Node: 
  def __init__(self, val):
    self.val = val
    self.next = None

def merge_lists(head_1, head_2):
  dummy_head = Node(None)
  tail = dummy_head
  current_1 = head_1
  current_2 = head_2
  
  while current_1 is not None and current_2 is not None:
    if current_1.val < current_2.val:
      tail.next = current_1
      current_1 = current_1.next
    else:
      tail.next = current_2
      current_2 = current_2.next
    tail = tail.next
    
  if current_1 is not None:
      tail.next = current_1
    
  if current_2 is not None:
      tail.next = current_2
  
  return dummy_head.next

# n = length of list 1
# m = length of list 2
# Time: O(min(n, m))
# Space: O(1)

# recursive

def merge_lists(head_1, head_2):
  if head_1 is None and head_2 is None:
    return None
  if head_1 is None:
    return head_2
  if head_2 is None:
    return head_1
  if head_1.val < head_2.val:
    next_1 = head_1.next
    head_1.next = merge_lists(next_1, head_2)
    return head_1
  else:
    next_2 = head_2.next
    head_2.next = merge_lists(head_1, next_2)
    return head_2
  
# n = length of list 1
# m = length of list 2
# Time: O(min(n, m))
# Space: O(min(n, m))


######################
## Is Univalue List ##
######################

# Iterative

class Node:
  def __init__(self, val):
    self.val = val
    self.next = None

def is_univalue_list(head):
  current = head
  while current is not None:
    if head.val != current.val:
      return False
    current = current.next 
  return True

# n = number of nodes
# Time: O(n)
# Space: O(1)

# Recursive 

def is_univalue_list(head, prev_val = None):
  if head is None:
    return True
  if prev_val is None or head.val == prev_val:
    return is_univalue_list(head.next, head.val)
  else:
    return False
  
# n = number of nodes
# Time: O(n)
# Space: O(n)


####################
## Longest Streak ##
####################

# Iterative

class Node:
  def __init__(self, val):
    self.val = val
    self.next = None

def longest_streak(head):
  max_streak = 0
  current_streak = 0
  prev_val = None
  
  current_node = head
  while current_node is not None:
    if current_node.val == prev_val:
      current_streak += 1
    else:
      current_streak = 1
  
    prev_val = current_node.val
    if current_streak > max_streak:
      max_streak = current_streak
    
    current_node = current_node.next
    
  return max_streak

# n = number of nodes
# Time: O(n)
# Space: O(1)


#################
## Remove Node ##
#################

# Iterative 

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def remove_node(head, target_val):
    if head.val == target_val:
        return head.next

    current = head
    prev = None
    while current is not None:
        if current.val == target_val:
            prev.next = current.next
            break

        prev = current
        current = current.next
    return head

# n = number of nodes
# Time: O(n)
# Space: O(1)

# Recursive

def remove_node(head, target_val):
  if head is None:
    return None

  if head.val == target_val:
    return head.next

  head.next = remove_node(head.next, target_val)
  return head

# n = number of nodes
# Time: O(n)
# Space: O(n)