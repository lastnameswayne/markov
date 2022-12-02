from tkinter import E
import numpy as np

def init_model():
    trans_probs = np.zeros([43,43])
    
    emission_probs = np.zeros([43,4])

    trans_probs[0,0]=0.99666
    trans_probs[0,1]=0.00144
    trans_probs[0,4]=0.00012
    trans_probs[0,7]=0.00015
    trans_probs[0,22]=0.00116
    trans_probs[0,25]=0.0003
    trans_probs[0,28]=0.00017
    
    trans_probs[1,2]=1
    trans_probs[4,5]=1
    trans_probs[7,8]=1
    trans_probs[22,23]=1
    trans_probs[25,26]=1
    trans_probs[28,29]=1

    trans_probs[2,3]=1
    trans_probs[5,6]=1
    trans_probs[8,9]=1
    trans_probs[23,24]=1
    trans_probs[26,27]=1
    trans_probs[29,30]=1

    trans_probs[3,10]=1
    trans_probs[6,10]=1
    trans_probs[9,10]=1
    trans_probs[24,31]=1
    trans_probs[27,31]=1
    trans_probs[27,30]=1

    trans_probs[10,11]=1
    trans_probs[31,32]=1

    trans_probs[12,10]=0.99886
    trans_probs[33,31]=0.99886

    trans_probs[12,13]=0.00021
    trans_probs[12,16]=0.00015
    trans_probs[12,19]=0.00078
    trans_probs[33,34]=0.00093
    trans_probs[33,37]=0.00009
    trans_probs[33,40]=0.00009

    trans_probs[13,14]=1
    trans_probs[16,17]=1
    trans_probs[19,20]=1
    trans_probs[34,35]=1
    trans_probs[37,38]=1
    trans_probs[40,41]=1

    trans_probs[14,15]=1
    trans_probs[17,18]=1
    trans_probs[20,21]=1
    trans_probs[35,36]=1
    trans_probs[38,39]=1
    trans_probs[41,42]=1

    trans_probs[15,0]=1
    trans_probs[18,0]=1
    trans_probs[21,0]=1
    trans_probs[36,0]=1
    trans_probs[39,0]=1
    trans_probs[42,0]=1

    emission_probs[0,0]=0.33434
    emission_probs[0,1]=0.15479
    emission_probs[0,2]=0.16613
    emission_probs[0,3]=0.33474

    emission_probs[1,0]=1
    emission_probs[4,2]=1
    emission_probs[7,3]=1
    emission_probs[22,3]=1
    emission_probs[25,1]=1
    emission_probs[28,3]=1

    emission_probs[2,3]=1
    emission_probs[5,3]=1
    emission_probs[8,3]=1
    emission_probs[23,3]=1
    emission_probs[26,3]=1
    emission_probs[29,1]=1

    emission_probs[3,2]=1
    emission_probs[6,2]=1
    emission_probs[9,2]=1
    emission_probs[24,0]=1
    emission_probs[27,0]=1
    emission_probs[30,0]=1

    emission_probs[10,0]=0.32121
    emission_probs[10,1]=0.15903
    emission_probs[10,2]=0.32265
    emission_probs[10,3]=0.19712

    emission_probs[31,0]=0.39946
    emission_probs[31,1]=0.13241
    emission_probs[31,2]=0.12861
    emission_probs[31,3]=0.33952

    emission_probs[11,0]=0.35257
    emission_probs[11,1]=0.20005
    emission_probs[11,2]=0.13622
    emission_probs[11,3]=0.3115

    emission_probs[32,0]=0.31271
    emission_probs[32,1]=0.13711
    emission_probs[32,2]=0.19756
    emission_probs[32,3]=0.35261

    emission_probs[12,0]=0.33922
    emission_probs[12,1]=0.12998
    emission_probs[12,2]=0.13122
    emission_probs[12,3]=0.39958

    emission_probs[33,0]=0.19905
    emission_probs[33,1]=0.31826
    emission_probs[33,2]=0.16051
    emission_probs[33,3]=0.32219

    emission_probs[13,3]=1
    emission_probs[16,3]=1
    emission_probs[19,3]=1
    emission_probs[34,1]=1
    emission_probs[37,1]=1
    emission_probs[40,1]=1

    emission_probs[14,0]=1
    emission_probs[17,2]=1
    emission_probs[20,0]=1
    emission_probs[35,0]=1
    emission_probs[38,0]=1
    emission_probs[41,0]=1

    emission_probs[15,2]=1
    emission_probs[18,0]=1
    emission_probs[21,0]=1
    emission_probs[36,3]=1
    emission_probs[39,1]=1
    emission_probs[42,0]=1

    init_probs = np.zeros((43))
    init_probs[0]=1.00

    return hmm(init_probs, trans_probs, emission_probs)

def translate_path_to_indices(path):
    return list(map(lambda x: int(x), path))

def translate_indices_to_path(indices):
    mapping = np.empty((42), dtype='object')
    mapping[0]='N'
    mapping[1:21]='C'
    mapping[22:42]='R'
    print(mapping.shape)
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
    for n in range(N-2, -1, -1):
        z[n] = oldZ[int(z[n+1]), n]
    return z

def compute_w_log(model, x):
    k = len(model.init_probs)
    trans_probs = model.trans_probs
    emission_probs = model.emission_probs
    init_probs = np.array(model.init_probs)
    n = len(x)
    w = make_table(k, n)
    w=np.zeros([k,n])
    trans_probs=np.array(trans_probs)
    emission_probs=np.array(emission_probs)
    
    z = np.zeros((k, n-1)).astype(np.int32)
    for i in range(k):
        w[i,0]=np.log(init_probs[i])+np.log(emission_probs[i, x[0]])    
    for i in range(1,n):
        for j in range(k):
            #third = w[j][i-1]
            #first = w[j][i]
            #second = emission_probs[j,x[i]]
            #fourth = trans_probs[:,j]
            #w[j,i]=max(first, second)*third*fourth
            prob = w[:,i-1] + np.log(trans_probs[:,j])
            w[j,i] = max(prob)+np.log(emission_probs[j,x[i]])
            z[j,i-1]=np.argmax(prob)
            
            #probability = w[i - 1] + np.log(trans_probs[j,:]) + np.log(emission_probs[j, x[i]])
            #w[j,i] = max(probability)
    return w,z

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
    
    xArr = translate_observations_to_indices(x)
    zArr = list(z)
    for idx in range(len(xArr)-1):
        i = int(zArr[idx])
        j = int(zArr[idx+1])
        
        symbol = xArr[idx]
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
    init_probs = np.zeros((43))
    init_probs[0]=1.00

    for i in range(len(counts_transitions)):
        for j in range(len(counts_transitions[0])):
            total_to_itself = counts_transitions[i][j]
            total_to_others = sum(counts_transitions[i])
            prob = total_to_itself/total_to_others
            prob_matrix_transitions[i][j]=prob
            
            
    for i in range(len(counts_symbol)):
        for j in range(len(counts_symbol[0])):
            total_to_itself = counts_symbol[i][j]
            total_to_others = sum(counts_symbol[i])
            prob = total_to_itself/total_to_others
            prob_matrix_emissions[i][j]=prob
        

    
    return hmm(init_probs, prob_matrix_transitions, prob_matrix_emissions)



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


def train(model, x, z):
    x = translate_observations_to_indices(x)
    z = translate_path_to_indices(x)
    for i in range(50):
        model = viterbi_update_model(model, x,z,43,4)
        print(model.trans_probs)
    return model
        
    

g1 = read_fasta_file('genome1.fa')
g1_true = read_fasta_file('true-ann1.fa')

x=g1['genome1']
trueAnnotations=(g1_true['true-ann1'])
myModel = init_model()

# Your code to read the annotations and compute the accuracies of your predictions...
def compute_accuracy(true_ann, pred_ann):
    if len(true_ann) != len(pred_ann):
        return 0.0
    return sum(1 if true_ann[i] == pred_ann[i] else 0 
               for i in range(len(true_ann))) / len(true_ann)

g1 = read_fasta_file('genome1.fa')
g1_true = read_fasta_file('true-ann1.fa')
x=g1['genome1'][:100]
true = g1_true['true-ann1'][:100]
w,z = compute_w_log(myModel, x)
z_viterbi = backtrack(myModel, x, w, z)
print(z_viterbi)
print("")
print(true)
print("acc",compute_accuracy(true,translate_indices_to_path(z_viterbi)))


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


#final_hmm = main(7,4, trueAnnotations, model)

