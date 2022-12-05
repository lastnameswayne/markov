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
    init_probs[0]=1
    print("init_probs", init_probs)

    return hmm(init_probs, trans_probs, emission_probs)
class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

