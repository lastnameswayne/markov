from re import L
from tkinter import E
import numpy as np

def init_model():
    init_probs_43_state = np.zeros(43, dtype = float)
    init_probs_43_state[0] = 1.0

    trans_probs_43_state = np.zeros((43, 43), dtype = float)
    trans_probs_43_state[0,  0] = 0.99666   #Staying in the non-coding state
    trans_probs_43_state[0,  1] = 0.0144    #ATG start codon — Probably too likely
    trans_probs_43_state[1,  2] = 1.0
    trans_probs_43_state[2,  3] = 1.0
    trans_probs_43_state[3,  10] = 1.0
    trans_probs_43_state[0,  4] = 0.0012    #GTG start codon
    trans_probs_43_state[4,  5] = 1.0
    trans_probs_43_state[5,  6] = 1.0
    trans_probs_43_state[6,  10] = 1.0
    trans_probs_43_state[0,  7] = 0.0015    #TTG start codon
    trans_probs_43_state[7,  8] = 1.0
    trans_probs_43_state[8,  9] = 1.0
    trans_probs_43_state[9,  10] = 1.0
    trans_probs_43_state[0, 22] = 0.0166    #TTA reverse start codon
    trans_probs_43_state[22, 23] = 1.0
    trans_probs_43_state[23, 24] = 1.0
    trans_probs_43_state[24, 31] = 1.0
    trans_probs_43_state[0, 25] = 0.003     #CTA reverse start codon
    trans_probs_43_state[25, 26] = 1.0
    trans_probs_43_state[26, 27] = 1.0
    trans_probs_43_state[27, 31] = 1.0
    trans_probs_43_state[0, 28] = 0.0017    #TCA reverse start codon
    trans_probs_43_state[28, 29] = 1.0
    trans_probs_43_state[29, 30] = 1.0
    trans_probs_43_state[30, 31] = 1.0
    trans_probs_43_state[10, 11] = 1.0      #Forward internal codon
    trans_probs_43_state[11, 12] = 1.0
    trans_probs_43_state[12, 10] = 0.99886
    trans_probs_43_state[31, 32] = 1.0      #Reverse internal codon
    trans_probs_43_state[32, 33] = 1.0
    trans_probs_43_state[33, 31] = 0.99889
    trans_probs_43_state[12, 13] = 0.0021      #TAG stop codon
    trans_probs_43_state[13, 14] = 1.0
    trans_probs_43_state[14, 15] = 1.0
    trans_probs_43_state[15, 0] = 1.0
    trans_probs_43_state[12, 16] = 0.0015      #TGA stop codon
    trans_probs_43_state[16, 17] = 1.0
    trans_probs_43_state[17, 18] = 1.0
    trans_probs_43_state[18, 0] = 1.0
    trans_probs_43_state[12, 19] = 0.0015      #TAA stop codon
    trans_probs_43_state[19, 20] = 1.0
    trans_probs_43_state[20, 21] = 1.0
    trans_probs_43_state[21, 0] = 1.0
    trans_probs_43_state[33, 34] = 0.0093      #CAT reverse stop codon
    trans_probs_43_state[34, 35] = 1.0
    trans_probs_43_state[35, 36] = 1.0
    trans_probs_43_state[36, 0] = 1.0
    trans_probs_43_state[33, 37] = 0.0009      #CAC reverse stop codon
    trans_probs_43_state[37, 38] = 1.0
    trans_probs_43_state[38, 39] = 1.0
    trans_probs_43_state[39, 0] = 1.0
    trans_probs_43_state[33, 40] = 0.0009      #CAA reverse stop codon
    trans_probs_43_state[40, 41] = 1.0
    trans_probs_43_state[41, 42] = 1.0
    trans_probs_43_state[42, 0] = 1.0

    #ACGT
    emission_probs_43_state = np.zeros((43, 4), dtype = float)
    emission_probs_43_state[0] = [0.33434, 0.16479, 0.16613, 0.33474]
    emission_probs_43_state[1] = [1.0, 0.0, 0.0, 0.0]   #ATG start codon
    emission_probs_43_state[2] = [0.0, 0.0, 0.0, 1.0]
    emission_probs_43_state[3] = [0.0, 0.0, 1.0, 0.0]
    emission_probs_43_state[4] = [0.0, 0.0, 1.0, 0.0]   #GTG start codon
    emission_probs_43_state[5] = [0.0, 0.0, 0.0, 1.0]
    emission_probs_43_state[6] = [0.0, 0.0, 1.0, 0.0]
    emission_probs_43_state[7] = [0.0, 0.0, 0.0, 1.0]   #TTG start codon
    emission_probs_43_state[8] = [0.0, 0.0, 0.0, 1.0]
    emission_probs_43_state[9] = [0.0, 0.0, 1.0, 0.0]
    emission_probs_43_state[10] = [0.32121, 0.15903, 0.32256, 0.19712]  #Forward internal codon
    emission_probs_43_state[11] = [0.35257, 0.20005, 0.13622, 0.31115]
    emission_probs_43_state[12] = [0.33922, 0.12998, 0.13122, 0.39958]
    emission_probs_43_state[13] = [0.0, 0.0, 0.0, 1.0]   #TAG stop codon
    emission_probs_43_state[14] = [1.0, 0.0, 0.0, 0.0]
    emission_probs_43_state[15] = [0.0, 0.0, 1.0, 0.0]
    emission_probs_43_state[16] = [0.0, 0.0, 0.0, 1.0]   #TGA stop codon
    emission_probs_43_state[17] = [0.0, 0.0, 1.0, 0.0]
    emission_probs_43_state[18] = [1.0, 0.0, 0.0, 0.0]
    emission_probs_43_state[19] = [0.0, 0.0, 0.0, 1.0]   #TAA stop codon
    emission_probs_43_state[20] = [1.0, 0.0, 0.0, 0.0]
    emission_probs_43_state[21] = [1.0, 0.0, 0.0, 0.0]
    emission_probs_43_state[22] = [0.0, 0.0, 0.0, 1.0]   #TTA reverse start codon
    emission_probs_43_state[23] = [0.0, 0.0, 0.0, 1.0]
    emission_probs_43_state[24] = [1.0, 0.0, 0.0, 0.0]
    emission_probs_43_state[25] = [0.0, 1.0, 0.0, 0.0]   #CTA reverse start codon
    emission_probs_43_state[26] = [0.0, 0.0, 0.0, 1.0]
    emission_probs_43_state[27] = [1.0, 0.0, 0.0, 0.0]
    emission_probs_43_state[28] = [0.0, 0.0, 0.0, 1.0]   #TCA reverse start codon
    emission_probs_43_state[29] = [0.0, 1.0, 0.0, 0.0]
    emission_probs_43_state[30] = [1.0, 0.0, 0.0, 0.0]
    emission_probs_43_state[31] = [0.39946, 0.13241, 0.12861, 0.33952]  #Reverse internal codon
    emission_probs_43_state[32] = [0.31271, 0.13711, 0.19756, 0.35261]
    emission_probs_43_state[33] = [0.19905, 0.31826, 0.16051, 0.32219]
    emission_probs_43_state[34] = [0.0, 1.0, 0.0, 0.0]   #CAT reverse stop codon
    emission_probs_43_state[35] = [1.0, 0.0, 0.0, 0.0]
    emission_probs_43_state[36] = [0.0, 0.0, 0.0, 1.0]
    emission_probs_43_state[37] = [0.0, 1.0, 0.0, 0.0]   #CAC reverse stop codon
    emission_probs_43_state[38] = [1.0, 0.0, 0.0, 0.0]
    emission_probs_43_state[39] = [0.0, 1.0, 0.0, 0.0]
    emission_probs_43_state[40] = [0.0, 1.0, 0.0, 0.0]   #CAA reverse stop codon
    emission_probs_43_state[41] = [1.0, 0.0, 0.0, 0.0]
    emission_probs_43_state[42] = [1.0, 0.0, 0.0, 0.0]


    return hmm(init_probs_43_state, trans_probs_43_state, emission_probs_43_state)

def init_7_state():
    init_probs_7_state = np.array([0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00])

    trans_probs_7_state = np.array([
        [0.00, 0.00, 0.90, 0.10, 0.00, 0.00, 0.00],
        [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
        [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00],
    ])

    emission_probs_7_state = np.array([
        #   A     C     G     T
        [0.30, 0.25, 0.25, 0.20],
        [0.20, 0.35, 0.15, 0.30],
        [0.40, 0.15, 0.20, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.20, 0.40, 0.30, 0.10],
        [0.30, 0.20, 0.30, 0.20],
        [0.15, 0.30, 0.20, 0.35],
    ])

    return hmm(init_probs_7_state, trans_probs_7_state, emission_probs_7_state)

def translate_path_to_indices(path):
    return list(map(lambda x: int(x), path))


def translate_annotation_to_hidden_state_7_state(path):
    pathArr = list(path)
    path=pathArr
    hidden = np.zeros((len(path)), dtype="int")
    for i in range(len(path)):
        if (path[i]=="N"):
            hidden[i]= 3
        elif (path[i]=="C"):
            if (hidden[i-1]==3):
                hidden[i]=2
            elif (hidden[i-1]==0):
                hidden[i]=2
            elif(hidden[i-1]==2):
                hidden[i]=1
            elif(hidden[i-1]==1):
                hidden[i]=0
        elif (path[i]=="R"):
            if (hidden[i-1]==3):
                hidden[i]=4
            elif (hidden[i-1]==6):
                hidden[i]=4
            elif(hidden[i-1]==4):
                hidden[i]=5
            elif(hidden[i-1]==5):
                hidden[i]=6
    return hidden

def translate_indices_to_path_7_state(indices):
    mapping = np.empty((43), dtype='object')
    mapping = ['C', 'C', 'C', 'N', 'R', 'R', 'R']
    return ''.join([mapping[i] for i in indices])


def translate_indices_to_path(indices):
    mapping = np.empty((43), dtype='object')
    mapping[0]='N'
    mapping[1:21]='C'
    mapping[21:43]='R'
    #mapping = ['C', 'C', 'C', 'N', 'R', 'R', 'R']
    return ''.join([mapping[i] for i in indices])

def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs


def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]

def joint_prob_log(model, x, z):
    trans_probs = model.trans_probs
    emission_probs = model.emission_probs
    init_probs = model.init_probs
    logp = np.log(init_probs[z[0]]) + np.log(emission_probs[z[0]][x[0]])
    for i in range(1, len(x)):
        logp = logp + np.log(trans_probs[z[i-1]][z[i]]) + np.log(emission_probs[z[i]][x[i]])
    return logp

def opt_path_prob_log(w):
    last_col = w[:,-1]
    return np.log(np.amax(last_col))

def backtrack(model, x, w,oldZ):
    x = np.array(x)
    trans_probs = np.array(model.trans_probs)
    emission_probs = np.array(model.emission_probs)
    N = w.shape[1]
    z=np.zeros(N, dtype=int)
    z[N-1] = np.argmax(w[:,-1])
    print("old",oldZ, oldZ.shape)
    for n in range(N-2, -1, -1):
        z[n] = oldZ[int(z[n+1]), n]
    return z

def compute_w_log(model, x):
    k = len(model.init_probs)
    trans_probs = model.trans_probs
    emission_probs = model.emission_probs
    init_probs = np.array(model.init_probs)
    n = len(x)
    w=np.zeros([k,n])
    print(k,n)
    trans_probs=np.array(trans_probs)
    emission_probs=np.array(emission_probs)
    z = np.zeros((k, n-1)).astype(np.int32)
    w[:, 0] = init_probs * emission_probs[:, x[0]]  
    for i in range(1,n):
        for j in range(k):
            prob = w[:,i-1] + np.log(trans_probs[:,j])
            w[j,i] = max(prob)+np.log(emission_probs[j,x[i]])
            z[j,i-1]=np.argmax(prob)
    return w,z


def viterbi_log(self, x):
    K = self.init_probs.shape[0] 
    # number of states N = Len(x) # length of observation sequence
    # An indice in the omega matrix k, n represents the probability of being in state k at step n # first column of omega matrix is the start probabilities of the states times their emission probability of the first observation omega = np.ones((K, N)) (np.inf) #k rows n columns
    N = len(x)
    omega = np.ones((K,N))*(-np.inf)
    omega[:,0] = np.log(self.init_probs) + np.log(self.emission_probs[:, x[0]])
    for n in range(1, N):
        for k in range(K):
            omega[k, n] = np.max(
                omega[:, n - 1] + np.log(self.trans_probs[:, k]) + np.log(self.emission_probs[k, x[n]]))
    #backtracking
    Z = np.zeros(N, dtype=int)
    Z[-1] = np.argmax(omega[:, -1])
    for n in range (N - 2, -1, -1):
        Z[n] = np.argmax(
            np.log(self.emission_probs [Z[n + 1], x[n + 1]])+ omega[:, n] + np.log(self.trans_probs [:, Z[n + 1]]))
    return Z

def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences

def count_transitions_and_emissions(K, D, x, z):
    """
    Returns a KxK matrix and a KxD matrix containing counts cf. above
    """
    counts_transitions = make_table(K,K)
    counts_symbol = make_table(K,D)

    for idx in range(len(x)-1):
        i = int(z[idx])
        j = int(z[idx+1])
        
        symbol = x[idx]
        counts_transitions[i][j] = counts_transitions[i][j]+1
        counts_symbol[i][symbol]=counts_symbol[i][symbol]+1
                
    return counts_transitions, counts_symbol

def training_by_counting(K, D, x, z):
    """
    Returns a HMM trained on x and z cf. training-by-counting.
    """
    counts_transitions, counts_symbol = count_transitions_and_emissions(K,D, x, z)
    prob_matrix_transitions = make_table(K,K)
    prob_matrix_emissions = make_table(K,D)
    init_probs = np.zeros((K))
    init_probs[3]=1.00

    for i in range(len(counts_transitions)):
        for j in range(len(counts_transitions[0])):
            total_to_itself = counts_transitions[i][j]
            total_to_others = sum(counts_transitions[i])
            if (total_to_others != 0):
                prob = total_to_itself/total_to_others
                prob_matrix_transitions[i][j]=prob
            
            
    for i in range(len(counts_symbol)):
        for j in range(len(counts_symbol[0])):
            total_to_itself = counts_symbol[i][j]
            total_to_others = sum(counts_symbol[i])
            if (total_to_others != 0):
                prob = total_to_itself/total_to_others
                prob_matrix_emissions[i][j]=prob
        

    
    return hmm(np.array(init_probs), np.array(prob_matrix_transitions), np.array(prob_matrix_emissions))




def viterbi_update_model(model, x, K, D):
    """
    return a new model that corresponds to one round of Viterbi training, 
    i.e. a model where the parameters reflect training by counting on x 
    and z_vit, where z_vit is the Viterbi decoding of x under the given 
    model.
    """
    
    w, oldZ = compute_w_log(model, x)
    z = backtrack(model, x, w, oldZ)
    new_model = training_by_counting(K, D, x, z)
    return new_model


def train_to_convergence(model, x, z):
    firstModel = training_by_counting(model.init_probs.shape[0], model.emission_probs.shape[1], x,z)
    old_model = firstModel
    iteration = 0
    #initialTraining
    while True:
        print("starting iteration", iteration)
        new_model = viterbi_update_model(old_model, x, model.init_probs.shape[0], model.emission_probs.shape[1])

        trans_prob_cmp = abs(new_model.trans_probs - old_model.trans_probs) < 0.01
        emission_prob_cmp = abs(new_model.emission_probs - old_model.emission_probs) < 0.01
        init_prob_cmp = abs(new_model.init_probs - old_model.init_probs) < 0.01
        print("trans_probs",new_model.trans_probs)
        print("emit_probs", new_model.emission_probs)
        if (np.all(trans_prob_cmp) and np.all(emission_prob_cmp) and np.all(init_prob_cmp)) or iteration>100:
            print("Converged  at iteration " + str(iteration))
            break

        old_model = new_model
        iteration += 1
    return new_model
        

def compute_accuracy(true_ann, pred_ann):
    if len(true_ann) != len(pred_ann):
        return 0.0
    return sum(1 if true_ann[i] == pred_ann[i] else 0 
               for i in range(len(true_ann))) / len(true_ann) 



# x1=g1['genome1']
# trueAnnotations1=(g1_true['true-ann1'])
myModel = init_7_state()
print("before trans_probs",myModel.trans_probs)
print("before emission probs", myModel.emission_probs)
# Your code to read the annotations and compute the accuracies of your predictions...


# g1 = read_fasta_file('genome1.fa')
# g1_true = read_fasta_file('true-ann1.fa')
# x=g1['genome1'][:10000]
# true = g1_true['true-ann1'][:10000]
# x_indices=translate_observations_to_indices(x)
# training_by_counting(43,4, x_indices,true, x)
# w,z = compute_w_log(myModel, x)
# z_viterbi = backtrack(myModel, x, w, z)
# print("acc",compute_accuracy(true,translate_indices_to_path(z_viterbi)))

# print("starting training")
# myTrainedModel = train_to_convergence(myModel, x)
# w_trained,z_trained = compute_w_log(myTrainedModel, x)
# z_viterbi_trained = backtrack(myTrainedModel, x, w_trained, z_trained)
# print("acc",compute_accuracy(true,translate_indices_to_path(z_viterbi_trained)))

# finalS = ''.join(translate_indices_to_path(z_viterbi))



def trainOnAll5FindBestModel():
    #my known genomes
    g1 = read_fasta_file('genome1.fa').get('genome1')
    g1_true = read_fasta_file('true-ann1.fa').get('true-ann1')

    g2 = read_fasta_file('genome2.fa').get('genome2')
    g2_true = read_fasta_file('true-ann2.fa').get('true-ann2')
    g3 = read_fasta_file('genome3.fa').get('genome3')
    g3_true = read_fasta_file('true-ann3.fa').get('true-ann3')
    g4 = read_fasta_file('genome4.fa').get('genome4')
    g4_true = read_fasta_file('true-ann4.fa').get('true-ann4')
    g5 = read_fasta_file('genome5.fa').get('genome5')
    g5_true = read_fasta_file('true-ann5.fa').get('true-ann5')
    
    x_1=translate_observations_to_indices(g1)
    x_2=translate_observations_to_indices(g2)
    x_3=translate_observations_to_indices(g3)
    x_4=translate_observations_to_indices(g4)
    x_5=translate_observations_to_indices(g5)
    z_1 = translate_annotation_to_hidden_state_7_state(g1_true)
    z_2=translate_annotation_to_hidden_state_7_state(g2_true)
    z_3=translate_annotation_to_hidden_state_7_state(g3_true)
    z_4=translate_annotation_to_hidden_state_7_state(g4_true)
    z_5=translate_annotation_to_hidden_state_7_state(g5_true)
    
    print("starting training")
    myTrainedModel1 = train_to_convergence(myModel, x_1, z_1)
    myTrainedModel2 = train_to_convergence(myTrainedModel1, x_2, z_2)
    myTrainedModel3 = train_to_convergence(myTrainedModel2, x_3, z_3)
    myTrainedModel4 = train_to_convergence(myTrainedModel3, x_4, z_4)
    myTrainedModel5 = train_to_convergence(myTrainedModel4, x_5, z_5)

    print("DONE WITH TRAINING")

    #unknown genes
    g6 = read_fasta_file('genome6.fa')['genome6']
    g7 = read_fasta_file('genome7.fa')['genome7']
    g8 = read_fasta_file('genome8.fa')['genome8']
    g9 = read_fasta_file('genome9.fa')['genome9']
    g10 = read_fasta_file('genome10.fa')['genome10']
    
    x_6=translate_observations_to_indices(g6)
    w_6,z_6 = compute_w_log(myTrainedModel5, x_6)
    z_viterbi_6 = backtrack(myTrainedModel5, x_6, w_6, z_6)
    finalS = ''.join(translate_indices_to_path_7_state(z_viterbi_6))
    with open('predict6.fa', 'w') as fp:
        fp.write(finalS)

    x_7=translate_observations_to_indices(g7)
    w_7,z_7 = compute_w_log(myTrainedModel5, x_7)
    z_viterbi_7 = backtrack(myTrainedModel5, x_7, w_7, z_7)
    finalS = ''.join(translate_indices_to_path_7_state(z_viterbi_7))
    with open('predict7.fa', 'w') as fp:
        fp.write(finalS)

    x_8=translate_observations_to_indices(g8)
    w_8,z_8 = compute_w_log(myTrainedModel5, x_8)
    z_viterbi_8 = backtrack(myTrainedModel5, x_8, w_8, z_8)
    finalS = ''.join(translate_indices_to_path_7_state(z_viterbi_8))
    with open('predict8.fa', 'w') as fp:
        fp.write(finalS)


    x_9=translate_observations_to_indices(g9)
    w_9,z_9 = compute_w_log(myTrainedModel5, x_9)
    z_viterbi_9 = backtrack(myTrainedModel5, x_9, w_9, z_9)
    finalS = ''.join(translate_indices_to_path_7_state(z_viterbi_9))
    with open('predict9.fa', 'w') as fp:
        fp.write(finalS)

    x_10=translate_observations_to_indices(g10)
    w_10,z_10 = compute_w_log(myTrainedModel5, x_10)
    z_viterbi_10 = backtrack(myTrainedModel5, x_10, w_10, z_10)
    finalS = ''.join(translate_indices_to_path_7_state(z_viterbi_10))
    with open('predict10.fa', 'w') as fp:
        fp.write(finalS)

    print("DONE WITH PREDICTING")


def FiveFoldCrossValidation():
    #train on 1,2,3,4 , predict on 5
    g1 = read_fasta_file('genome1.fa')['genome1']
    g1_true = read_fasta_file('true-ann1.fa')['true-ann1']

    g2 = read_fasta_file('genome2.fa')['genome2']
    g2_true = read_fasta_file('true-ann2.fa')['true-ann2']

    g3 = read_fasta_file('genome3.fa')['genome3']
    g3_true = read_fasta_file('true-ann3.fa')['true-ann3']

    g4 = read_fasta_file('genome4.fa')['genome4']
    g4_true = read_fasta_file('true-ann4.fa')['true-ann4']

    g5 = read_fasta_file('genome5.fa')['genome5']
    g5_true = read_fasta_file('true-ann5.fa')['true-ann5']
    
    x_1=translate_observations_to_indices(g1)
    x_2=translate_observations_to_indices(g2)
    x_3=translate_observations_to_indices(g3)
    x_4=translate_observations_to_indices(g4)
    x_5=translate_observations_to_indices(g5)
    z_1 = translate_annotation_to_hidden_state_7_state(g1_true)
    z_2=translate_annotation_to_hidden_state_7_state(g2_true)
    z_3=translate_annotation_to_hidden_state_7_state(g3_true)
    z_4=translate_annotation_to_hidden_state_7_state(g4_true)
    z_5=translate_annotation_to_hidden_state_7_state(g5_true)

    # #ROUND 1, train on 2,3,4,5
    # myTrainedModel2 = train_to_convergence(myModel, x_2, z_2)
    # myTrainedModel3 = train_to_convergence(myTrainedModel2, x_3, z_3)
    # myTrainedModel4 = train_to_convergence(myTrainedModel3, x_4, z_4)
    # myTrainedModel5 = train_to_convergence(myTrainedModel4, x_5, z_5)
    # #ROUND 1, predict on 1
    # w_1,z_w_log_1 = compute_w_log(myTrainedModel5, x_1)
    # z_viterbi_1 = backtrack(myTrainedModel5, x_1, w_1, z_w_log_1)
    # finalS = ''.join(translate_indices_to_path_7_state(z_viterbi_1))
    # with open('predict1.fa', 'w') as fp:
    #     fp.write(finalS)

    # print("DONE WITH ROUND 1")

    #ROUND 2, train on 1,3,4,5
    print("STARTING ROUND 2")
    myTrainedModel1 = train_to_convergence(myModel, x_1, z_1)
    myTrainedModel3 = train_to_convergence(myTrainedModel1, x_3, z_3)
    myTrainedModel4 = train_to_convergence(myTrainedModel3, x_4, z_4)
    myTrainedModel5 = train_to_convergence(myTrainedModel4, x_5, z_5)
    #ROUND 2, predict on 2
    w_2,z_w_log_2 = compute_w_log(myTrainedModel5, x_2)
    z_viterbi_2 = backtrack(myTrainedModel5, x_2, w_2, z_w_log_2)
    finalS = ''.join(translate_indices_to_path_7_state(z_viterbi_2))
    with open('predict2.fa', 'w') as fp:
        fp.write(finalS)

    print("DONE WITH ROUND 2")

    #ROUND 3, train on 1,2,4,5
    myTrainedModel1 = train_to_convergence(myModel, x_1, z_1)
    myTrainedModel2 = train_to_convergence(myTrainedModel1, x_2, z_2)
    myTrainedModel4 = train_to_convergence(myTrainedModel2, x_4, z_4)
    myTrainedModel5 = train_to_convergence(myTrainedModel4, x_5, z_5)
    #ROUND 3, predict on 3
    w_3,z_w_log_3 = compute_w_log(myTrainedModel5, x_3)
    z_viterbi_3 = backtrack(myTrainedModel5, x_3, w_3, z_w_log_3)
    finalS = ''.join(translate_indices_to_path_7_state(z_viterbi_3))
    with open('predict3.fa', 'w') as fp:
        fp.write(finalS)
    print("DONE WITH ROUND 3")


    #ROUND 4, train on 1,2,3,5
    myTrainedModel1 = train_to_convergence(myModel, x_1, z_1)
    myTrainedModel2 = train_to_convergence(myTrainedModel1, x_2, z_2)
    myTrainedModel3 = train_to_convergence(myTrainedModel2, x_3, z_3)
    myTrainedModel5 = train_to_convergence(myTrainedModel3, x_5, z_5)
    #ROUND 4, predict on 4
    w_4,z_w_log_4 = compute_w_log(myTrainedModel5, x_4)
    z_viterbi_4 = backtrack(myTrainedModel5, x_4, w_4, z_w_log_4)
    finalS = ''.join(translate_indices_to_path_7_state(z_viterbi_4))
    with open('predict4.fa', 'w') as fp:
        fp.write(finalS)
    print("DONE WITH ROUND 4")

    #ROUND 5, train on 1,2,3,4
    myTrainedModel1 = train_to_convergence(myModel, x_1, z_1)
    myTrainedModel2 = train_to_convergence(myTrainedModel1, x_2, z_2)
    myTrainedModel3 = train_to_convergence(myTrainedModel2, x_3, z_3)
    myTrainedModel4 = train_to_convergence(myTrainedModel3, x_4, z_4)
    #ROUND 5, predict on 5
    w_5,z_w_log_5 = compute_w_log(myTrainedModel4, x_5)
    z_viterbi_5 = backtrack(myTrainedModel4, x_5, w_5, z_w_log_5)
    finalS = ''.join(translate_indices_to_path_7_state(z_viterbi_5))
    with open('predict5.fa', 'w') as fp:
        fp.write(finalS)

#trainOnAll5FindBestModel()
FiveFoldCrossValidation()
#decide on model




# def main(K, D, trueAnnotations, model):
#     w = compute_w_log(model, trueAnnotations)
#     z = backtrack_log(model, x, w)
#     for i in range(50):
#         final_hmm = viterbi_update_model(final_hmm, x, K, D)

#     return final_hmm

#decide on K and D??
#find initial probabilities and model
#initialisere en model?


#x is genome1
#use viterbi decoding to find the most likely hidden states from the model og x
#brug den viterbi decoding til at train by counting
#kør training by counting indtil modellens parametre ikke rykker sig længere
#brug den model til at sammenligne 
#sammenlign accuracy
#iterate igen

#når det er færdigt så:
#5 runder:
#runde 1: traning by counting på genome 2,3,4,5
            # predict din gene structure af genome 1
            #compute the approximate correlation coefficient mellem genome1 og true-genome 1



#brug din bedste model til at preduct genome 6,7,8,9,10

