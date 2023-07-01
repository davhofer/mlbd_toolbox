"""
bayesian knowledge tracing BKT
additive factors model AFM
performance factors analysis PFA
deep knowledge tracing (lecture 08)
"""

import random

STATE_MASTERED = 1
STATE_UNLEARNED = 0

class BKTModel:

    def __init__(self):
        self.p0 = 0
        self.pLearn = 0
        self.pForget = 0
        self.pGuess = 0
        self.pSlip = 0

        self.array_p_ot_otMinus1 = dict()
        self.array_p_st_otMinus1 = dict()
        self.array_p_st_ot = dict()

        self.outputs = []

    def set_state(self, p0, pLearn, pForget, pGuess, pSlip, outputs=[]):
        self.p0 = p0
        self.pLearn = pLearn
        self.pForget = pForget
        self.pGuess = pGuess
        self.pSlip = pSlip

        self.array_p_ot_otMinus1 = dict()
        self.array_p_st_otMinus1 = dict()
        self.array_p_st_ot = dict()

        self.outputs = outputs


    def get_p_st_otMinus1(self, t):
        if t in self.array_p_st_otMinus1.keys():
            return self.array_p_st_otMinus1[t]

        if t == 0:
            p = self.p0 
        else:
            p = (1 - self.pForget) * self.get_p_st_ot(t-1) + self.pLearn * (1 - self.get_p_st_ot(t-1))
        

        self.array_p_st_otMinus1[t] = p
        return p 

    def get_p_ot_otMinus1(self, t):
        if t in self.array_p_ot_otMinus1.keys():
            return self.array_p_ot_otMinus1[t]

        if t == 0:
            pstot = self.p0
        else:
            pstot = self.get_p_st_otMinus1(t)

        p = (1 - self.pSlip) * pstot + self.pGuess * (1-pstot)
        
        self.array_p_ot_otMinus1[t] = p
        return p
    """
        if self.outputs[t]:
            p = (1 - self.pSlip) * pstot + self.pGuess * (1-pstot)
        else:
            p = (self.pSlip) * pstot + (1-self.pGuess) * (1-pstot)
    """

    def get_p_st_ot(self, t):
        if t in self.array_p_st_ot.keys():
            return self.array_p_st_ot[t]

        if t == 0:
            pstot = self.p0
        else:
            pstot = self.get_p_st_otMinus1(t)


        if self.outputs[t]:
            p = (1 - self.pSlip) * pstot / ((1-self.pSlip) * pstot + self.pGuess*(1-pstot))
        else:
            p = (self.pSlip) * pstot / ((self.pSlip) * pstot + (1-self.pGuess)*(1-pstot))

        self.array_p_st_ot[t] = p
        return p


    def simulate_inference(self, T, probas=(0.5, 0.4, 0.0, 0.3, 0.2), outputs=None):
        self.set_state(*probas)
        if outputs:
            self.outputs = outputs 
        for t in range(T):
            print("t =", t)
            print(f"initial state proba: p(st|ot-1) = {self.get_p_st_otMinus1(t):.3f}")
            print(f"output proba: p(ot|ot-1) = {self.get_p_ot_otMinus1(t):.3f}")
            if len(self.outputs) > t:
                o = self.outputs[t]

                sampled = False
            else:
                o = 1 if random.random() < self.get_p_ot_otMinus1(t) else 0
                self.outputs.append(o)
                sampled = True

            print(f"actual output: {o} {'  (sampled)' if sampled else ''}")

            print(f"adjusted state proba: p(st|ot) = {self.get_p_st_ot(t):.3f}")
            print()

            

# TODO: learning parameters given input sequence

    def learn_parameters(self, sequences):
        pass


