import sys
import subprocess
import clingo
import numpy as np
import time
import os

# Configuration
fluentPrefix = 'fl_'
actionPrefix = 'act_'
observationPrefix = 'obs_'
action_project_file_name = 'tmp_action_project.lp'
fluent_project_file_name = 'tmp_fluent_project.lp'
observation_project_file_name = 'tmp_observation_project.lp'
state_obs_mapping_file_name = 'tmp_state_action_obs_mapping.lp'
predicateArityDivider = '$'
init_program = None

states = []
action2name = []
observations = []
transition_probs = None
transition_rwds = None
observation_probs = None
next_action_idx = 0

state_obs_definitions = ''

def runLPMLNProgram(ipt_file, args):
	cmd = 'lpmln2asp -i' + ipt_file + ' ' + args
	try:
		out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
	except Exception, e:
		if isinstance(e, OSError):
			out = str(e)
		else:
			out = str(e.output)
	return out

def getModelFromText(txt):
	#print txt
	model = []
	answers = txt.lstrip(' ').lstrip('\n').lstrip('\r')
	atoms = answers.split(' ')
	for atom in atoms:
		model.append(clingo.parse_term(atom))
	return model

def extractSpecialAtoms(answer_set, prefix):
	state = []
	for atom in answer_set:
		if atom.name.startswith(prefix):
			state.append(atom)
	return state

def findPredicateNamesWithPrefix(models, prefix):
	found = set([])
	for m in models:
		for atom in m:
			if atom.name.startswith(prefix):
				found.add(atom.name + "$" + str(len(atom.arguments)))
	return found

def createPredicateProjectFile(predicateSet, filename):
	out = open(filename, "w")
	for predicate in predicateSet:
		pred_name = predicate.split(predicateArityDivider)[0]
		arity = predicate.split(predicateArityDivider)[1]
		out.write('#show ' + pred_name + '/' + arity + '.\n')
	out.close()

def constructStates():
	rawOutput = runLPMLNProgram(program, '-all -clingo="-c m=0"')
	if 'UNSATISFIABLE' in rawOutput or "UNKNOWN" in rawOutput:
		print 'No action found. Exiting...'
		exit()
	rawAnswerSets = [x.split('\n')[1].lstrip(' ').lstrip('\n').lstrip('\r') for x in rawOutput.split('Answer: ')[1:]]
	answerSets = [getModelFromText(x) for x in rawAnswerSets]
	fluent_predicates = findPredicateNamesWithPrefix(answerSets, fluentPrefix)
	createPredicateProjectFile(fluent_predicates, fluent_project_file_name)
	rawOutput = runLPMLNProgram(program, '-e '+ fluent_project_file_name + ' -all -clingo="-c m=0 --project"')
	rawAnswerSets = [x.split('\n')[1].lstrip(' ').lstrip('\n').lstrip('\r') for x in rawOutput.split('Answer: ')[1:]]
	answerSets = [getModelFromText(x) for x in rawAnswerSets]
	state_descs = [extractSpecialAtoms(x, fluentPrefix) for x in answerSets]
	for desc in state_descs:
		states.append(desc)

def constructObservations():
	rawOutput = runLPMLNProgram(program, '-all -clingo="-c m=0"')
	if 'UNSATISFIABLE' in rawOutput or "UNKNOWN" in rawOutput:
		print 'No action found. Exiting...'
		exit()
	rawAnswerSets = [x.split('\n')[1].lstrip(' ').lstrip('\n').lstrip('\r') for x in rawOutput.split('Answer: ')[1:]]
	answerSets = [getModelFromText(x) for x in rawAnswerSets]
	observation_predicates = findPredicateNamesWithPrefix(answerSets, observationPrefix)
	createPredicateProjectFile(observation_predicates, observation_project_file_name)
	rawOutput = runLPMLNProgram(program, '-e '+ observation_project_file_name + ' -all -clingo="-c m=0 --project"')
	rawAnswerSets = [x.split('\n')[1].lstrip(' ').lstrip('\n').lstrip('\r') for x in rawOutput.split('Answer: ')[1:]]
	answerSets = [getModelFromText(x) for x in rawAnswerSets]
	observation_descs = [extractSpecialAtoms(x, observationPrefix) for x in answerSets]
	for desc in observation_descs:
		observations.append(desc)

def constructActions(act_file):
	global next_action_idx
	global action2name
	actions = []
	rawOutput = runLPMLNProgram(act_file, '-all -clingo="-c m=1"')
	if 'UNSATISFIABLE' in rawOutput or "UNKNOWN" in rawOutput:
		print 'No action found. Exiting...'
		exit()
	rawAnswerSets = [x.split('\n')[1].lstrip(' ').lstrip('\n').lstrip('\r') for x in rawOutput.split('Answer: ')[1:]]
	answerSets = [getModelFromText(x) for x in rawAnswerSets]
	print len(answerSets), 'answer sets found.'
	action_predicates = findPredicateNamesWithPrefix(answerSets, actionPrefix)
	createPredicateProjectFile(action_predicates, action_project_file_name)
	rawOutput = runLPMLNProgram(act_file, '-e '+ action_project_file_name + ' -all -clingo="-c m=1 --project"')
	rawAnswerSets = [x.split('\n')[1].lstrip(' ').lstrip('\n').lstrip('\r') for x in rawOutput.split('Answer: ')[1:]]
	answerSets = [getModelFromText(x) for x in rawAnswerSets]
	action_descs = [extractSpecialAtoms(x, actionPrefix) for x in answerSets]
	for desc in action_descs:
		actions.append(desc)
		next_action_idx += 1
	action2name += actions
	return actions

def model2conjunction(model):
	return ','.join([str(x) for x in model])

def setTimestep(model, timestep):
	new_model = []
	for atom in model:
		new_atom = clingo.Function(atom.name, atom.arguments[:-1] + [clingo.Number(timestep)])
		new_model.append(new_atom)
	return new_model

def extractProbs(rawOutput):
	txt = rawOutput.split('Optimization: ')[-1]
	probabilityTexts = [x.split('\n')[0] for x in txt.split('Probability of Answer ')[1:]]
	prob_dict = np.zeros(len(probabilityTexts) + 1)
	for p in probabilityTexts:
		idx = int(p.split(' ')[0].lstrip('(').rstrip(')'))
		prob = float(p.split(' : ')[1])
		prob_dict[idx] = prob
	return prob_dict

def extractUtilityFromModel(model):
	utility = 0
	for atom in model:
		if atom.name == 'utility':
			utility += atom.arguments[0].number
	return utility

def extractTransitionInfo(actions, answerSets, prop_dict):
	global states
	global observations
	global transition_probs
	global transition_rwds
	global observation_probs
	transition_probs1 = np.zeros((len(actions),  len(states), len(states)))
	transition_rwds1 = np.zeros((len(actions), len(states), len(states)))
	observation_probs1 = np.zeros((len(actions), len(states), len(observations)))
	i = 1
	for a in answerSets:
		ss = -1
		es = -1
		act = -1
		obs = -1
		for atom in a:
			if atom.name == 'start_state':
				ss = atom.arguments[0].number
			elif atom.name == 'end_state':
				es = atom.arguments[0].number
			elif atom.name == 'action_idx':
				act = atom.arguments[0].number - len(actions)
			elif atom.name == 'obs_idx':
				obs = atom.arguments[0].number
		transition_probs1[act][ss][es] += prop_dict[i]
		observation_probs1[act][es][obs] += prop_dict[i]
		transition_rwds1[act][ss][es] = extractUtilityFromModel(a)
		i += 1
	# Normalize each column
	for ss in range(len(states)):
		for act in range(len(actions)):
			prob_sum = 0.0
			for es in range(len(states)):
				prob_sum += transition_probs1[act][ss][es]
			for es in range(len(states)):
				transition_probs1[act][ss][es] /= prob_sum
	for es in range(len(states)):
		for act in range(len(actions)):
			prob_sum = 0.0
			for obs in range(len(observations)):
				prob_sum += observation_probs1[act][es][obs]
			if prob_sum != 0.0:
				for obs in range(len(observations)):
					observation_probs1[act][es][obs] /= prob_sum
			else:
				observation_probs1[act][es][0] = 1.0
	transition_probs = np.concatenate((transition_probs, transition_probs1), axis=0)
	observation_probs = np.concatenate((observation_probs, observation_probs1), axis=0)
	transition_rwds = np.concatenate((transition_rwds, transition_rwds1), axis=0)

def makeTransitionsStochastic():
	global next_action_idx
	global transition_probs
	global observation_probs
	for a_idx in range(next_action_idx):
		for s_idx in range(len(states)):
			for i in range(len(states)-1, -1, -1):
				if transition_probs[a_idx][s_idx][i] != 0:
					transition_probs[a_idx][s_idx][i] = 1 - sum(transition_probs[a_idx][s_idx][:i])
					break
	for a_idx in range(next_action_idx):
		for s_idx in range(len(states)):
			for i in range(len(observations)-1, -1, -1):
				if observation_probs[a_idx][s_idx][i] != 0:
					observation_probs[a_idx][s_idx][i] = 1 - sum(observation_probs[a_idx][s_idx][:i])
					break


def createStateObservationDefinitions():
	global state_obs_definitions
	# Create definition for each transitions
	for s_idx in range(len(states)):
		state_obs_definitions += 'end_state(' + str(s_idx) + ') :- ' + model2conjunction(setTimestep(states[s_idx], 1)) + '.\n'
		state_obs_definitions += 'start_state(' + str(s_idx) + ') :- ' + model2conjunction(setTimestep(states[s_idx], 0)) + '.\n'
	for obs_idx in range(len(observations)):
		state_obs_definitions += 'obs_idx(' + str(obs_idx) + ') :- ' + model2conjunction(setTimestep(observations[obs_idx], 1)) + '.\n'
	#out = open(state_obs_mapping_file_name, 'w')
	#out.write(state_obs_definitions)
	#out.close()

def constructTransitionProbabilitiesAndTransitionRewardAndObservationProbabilities(act_file):
	actions = constructActions(act_file)
	print len(actions), 'actions detected from file', act_file, ': ', actions
	action_definitions = ''
	for a_idx in range(len(actions)):
		action_definitions += 'action_idx(' + str(a_idx)+ ') :- ' + model2conjunction(setTimestep(actions[a_idx], 0)) + '.\n'
	out = open(state_obs_mapping_file_name, 'w')
	out.write(state_obs_definitions + '\n')
	out.write(action_definitions)
	out.close()

	# Solve Tr(D, 1) once and collect output
	rawOutput = runLPMLNProgram(act_file, '-e '+ state_obs_mapping_file_name  + ' -all -clingo="-c m=1"')
	print 'Tr(D, 1) solving finished'
	rawAnswerSets = [x.split('\n')[1].lstrip(' ').lstrip('\n').lstrip('\r') for x in rawOutput.split('Answer: ')[1:]]
	answerSets = [getModelFromText(x) for x in rawAnswerSets]
	prob_dict = extractProbs(rawOutput)
	#print prob_dict
	extractTransitionInfo(actions, answerSets, prob_dict)
	#print transition_props
	print 'Tranisition Probabilities and Rewards extracted from', act_file, '.'

def createPOMDPFile():
	global action2name
	out_pomdp = open(pomdp_file_name, 'w')
	out_pomdp.write('# POMDP file for ' + program.split('.')[0] + '\n')
	out_pomdp.write('# generated by pbcplus2pomdp' + '\n')
	out_pomdp.write('# action ids: ' + '\n')
	for a_idx in range(next_action_idx):
		out_pomdp.write('# ' + str(a_idx) + ': ' + str(action2name[a_idx]) + '\n')
	out_pomdp.write('# state ids: ' + '\n')
	for s_idx in range(len(states)):
		out_pomdp.write('# ' + str(s_idx) + ': ' + str(states[s_idx]) + '\n')
	out_pomdp.write('# observation ids: ' + '\n')
	for o_idx in range(len(observations)):
		out_pomdp.write('# ' + str(o_idx) + ': ' + str(observations[o_idx]) + '\n')
	out_pomdp.write('\n\n')
	out_pomdp.write('discount: ' + str(discount) + '\n')
	out_pomdp.write('values: reward \n')
	out_pomdp.write('states: ' + str(len(states)) + '\n')
	out_pomdp.write('actions: ' + str(next_action_idx) + '\n')
	out_pomdp.write('observations: ' + str(len(observations)) + '\n')
	# Generate initial state probability distribution
	if init_program == None:
		out_pomdp.write('start: uniform \n')
	else:
		out_pomdp.write('start: \n')
		rawOutput = runLPMLNProgram(init_program, '-e '+ state_action_obs_mapping_file_name  + ' -all')
		print 'D_init solving finished'
		rawAnswerSets = [x.split('\n')[1].lstrip(' ').lstrip('\n').lstrip('\r') for x in rawOutput.split('Answer: ')[1:]]
		answerSets = [getModelFromText(x) for x in rawAnswerSets]
		prob_dict = extractProbs(rawOutput)
		init_probs = extractInitStateInfo(answerSets, prob_dict)
		print 'initial belief state:', init_probs
		for i in range(len(init_probs)):
			out_pomdp.write(str(init_probs[i]) + ' ')
	out_pomdp.write('\n')
	for a_idx in range(next_action_idx):
		for ss_idx in range(len(states)):
			for es_idx in range(len(states)):
				out_pomdp.write('T: ' + str(a_idx) + ' : ' + str(ss_idx) + ' : ' + str(es_idx) + ' ' + str(transition_probs[a_idx][ss_idx][es_idx]) + ' \n')
	out_pomdp.write('\n')
	for a_idx in range(next_action_idx):
		for es_idx in range(len(states)):
			for obs_idx in range(len(observations)):
				out_pomdp.write('O: ' + str(a_idx) + ' : ' + str(es_idx) + ' : ' + str(obs_idx) + ' ' + str(observation_probs[a_idx][es_idx][obs_idx]) + ' \n')
	out_pomdp.write('\n')
	for a_idx in range(next_action_idx):
		for ss_idx in range(len(states)):
			for es_idx in range(len(states)):
				out_pomdp.write('R: ' + str(a_idx) + ' : ' + str(ss_idx) + ' : ' + str(es_idx) + ' : *  ' + str(transition_rwds[a_idx][ss_idx][es_idx]) + ' \n')

	out_pomdp.close()

# Collect inputs
program = sys.argv[1]
action_dir = sys.argv[2]
#time_horizon = int(sys.argv[2])
discount = float(sys.argv[3])
pomdp_file_name = program.split('/')[-1].split('.')[0] + '.pomdp'
if len(sys.argv) >= 5:
	init_program = sys.argv[4]


start_time = time.time()
print 'Action Description in lpmln: ', program
#print 'Time Horizon: ', time_horizon
constructStates()
constructObservations()
print len(states), 'states detected: ', states
print len(observations), 'observations detected: ', observations
transition_probs = np.zeros((0,  len(states), len(states)))
transition_rwds = np.zeros((0, len(states), len(states)))
observation_probs = np.zeros((0, len(states), len(observations)))
for act_file in os.listdir(action_dir):
	if act_file.endswith('.lpmln'):
		print 'Processing file', act_file
		createStateObservationDefinitions()
		constructTransitionProbabilitiesAndTransitionRewardAndObservationProbabilities(os.path.join(action_dir, act_file))
print 'Making matrices stochastic...'
makeTransitionsStochastic()
print 'Transition Probabilitities: '
for a_idx in range(next_action_idx):
	print 'action ' + str(a_idx)
	print action2name[a_idx]
	print transition_probs[a_idx]
print 'Transition Rewards: '
for a_idx in range(next_action_idx):
	print 'action ' + str(a_idx)
	print action2name[a_idx]
	print transition_rwds[a_idx]
print 'Observation Probabilities: '
for a_idx in range(next_action_idx):
	print 'action ' + str(a_idx)
	print action2name[a_idx]
	print observation_probs[a_idx]
end_time = time.time()
lpmln_solving_time = end_time - start_time
createPOMDPFile()
print 'POMDP file saved as', pomdp_file_name
