import sys
import xml.etree.ElementTree as ET

action_vectors = {}
policy_file = sys.argv[1]

tree = ET.parse(policy_file)
root = tree.getroot()
alpha_vectors = root.findall("./AlphaVector/Vector")
for vec in alpha_vectors:
    act_id = int(vec.attrib['action'])
    vector = [float(x) for x in vec.text.split(' ') if len(x) > 0]
    if act_id in action_vectors:
        action_vectors[act_id].append(vector)
    else:
        action_vectors[act_id] = [vector]

#print action_vectors
numStates = len(action_vectors[1][0])
print 'Policy file parsing complete.'

while True:
    belief_str = raw_input('Enter belief state as single space separated vector, or "q" to quit: ')
    if len(belief_str) <= 0:
        continue
    if belief_str[0].lower() == 'q':
        exit()
    belief_vector = [float(x) for x in belief_str.split(' ')]
    if sum(belief_vector) != 1.0:
        print 'Belief state invalid: entries does not add up to one'
        continue
    if len(belief_vector) != numStates:
        print 'Belief state invalid: number of states does not match'
        continue
    print("belief state:", belief_vector)
    max_val = -float('inf')
    max_val_act = -1
    for act in action_vectors:
        for act_vec in action_vectors[act]:
            s = [act_vec[i] * belief_vector[i] for i in range(numStates)]
            s = sum(s)
            if s > max_val:
                max_val = s
                max_val_act = act
    print 'Best action:', max_val_act
