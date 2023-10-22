def prereqs_possible(num_courses, prereqs):
  graph = build_graph(num_courses, prereqs)
  visiting = set()
  visited = set()
  
  for node in range(0, num_courses):
    if has_cycle(graph, node, visiting, visited):
      return False
    
  return True
  
def has_cycle(graph, node, visiting, visited):
  if node in visited:
    return False
  
  if node in visiting:
    return True
  
  visiting.add(node)
  
  for neighbor in graph[node]:
    if has_cycle(graph, neighbor, visiting, visited):
      return True
  
  visiting.remove(node)
  visited.add(node)
  
  return False


def build_graph(num_courses, prereqs):
  graph = {}
  
  for i in range(0, num_courses):
    graph[i] = []
    
  for prereq in prereqs:
    a, b = prereq
    graph[a].append(b)
    
  return graph

# p = # prereqs
# n = # courses
# Time: O(n + p)
# Space: O(n)