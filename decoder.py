import argparse
import numpy as np
import pandas as pd

def forward(seq: str, emission: pd.DataFrame, p:float, q: float, p0: float, k: int) -> np.ndarray:
    """Calculate the k-th column of the forward table i.e.
       F_k = [P(X_1,...X_k | S=1), P(X_1,...X_k | S=2)]
       Note: by this definition, use np.logaddexp(forward(...)) with k=len(seq) to get the log likelihood

    Args:
        seq (str): The observations
        emission (pd.DataFrame): The emission table
        p (float): Prob to switch from state 1 to 2
        q (float): Prob to switch from state 2 to 1
        p0 (float): Prob to start in state 1
        k (int): The index k

    Returns:
        np.ndarray: [P(X_1,...X_k | S=1), P(X_1,...X_k | S=2)]
    """
    F_k = np.zeros((emission.shape[0], 2)) # A row for each state
    F_k[0,0] = np.log(p0 * emission.loc[0, seq[0]])
    F_k[1,0] = np.log((1 - p0)  * emission.loc[1, seq[0]])

    lnp = np.log(p)
    lnq = np.log(q)
    ln1p = np.log(1 - p)
    ln1q = np.log(1 - q)
    ln_emission = emission.copy()
    for c in ln_emission.columns:
        ln_emission[c] = np.log(ln_emission[c])

    for i in range(1, k):
        for j in range(F_k.shape[0]):
            tau = [ln1p, lnq] if j == 0 else [lnp, ln1q]
            tau = np.array(tau)

            ln_vals_to_sum = F_k[:,0] + tau + ln_emission.loc[j, seq[i]]
            F_k[j,1] = np.logaddexp(ln_vals_to_sum[0], ln_vals_to_sum[1])
        F_k[:,0] = F_k[:,1]
    
    return F_k[:,1]
    

def backward(seq: str, emission: pd.DataFrame, p:float, q: float, p0: float, k: int) -> np.ndarray:
    """Calculate the k-th element of the backward table i.e.
    B_k = [P(X_k+1,...X_n | S=1), P(X_k+1,...X_n | S=2)]

    Args:
        seq (str): The observations
        emission (pd.DataFrame): The emission table
        p (float): Prob to switch from state 1 to 2
        q (float): Prob to switch from state 2 to 1
        p0 (float): Prob to start in state 1
        k (int): The index k

    Returns:
        np.ndarray: [P(X_k+1,...X_n | S=1), P(X_k+1,...X_n | S=2)]
    """
    seq = '^' + seq
    B_k = np.zeros((emission.shape[0], 2)) # A row for each state
    lnp = np.log(p)
    lnq = np.log(q)
    ln1p = np.log(1 - p)
    ln1q = np.log(1 - q)
    lne = emission.copy()
    for c in lne.columns:
        lne[c] = np.log(lne[c])

    for i in range(len(seq) - 2, k - 1, -1):
        for j in range(B_k.shape[0]):
            if i > 0:
                tau = [ln1p, lnp] if j == 0 else [lnq, ln1q]
            else:
                tau = [np.log(p0), np.log(1 - p0)]
            tau = np.array(tau)

            ln_vals_to_sum = B_k[:,1] + tau + lne[seq[i + 1]].to_numpy()
            B_k[j,0] = np.logaddexp(ln_vals_to_sum[0], ln_vals_to_sum[1])
        B_k[:,1] = B_k[:,0]
    return B_k[:,0]


def posterior(seq: str, emission: pd.DataFrame, p:float, q: float, p0: float) -> str:
    post = []
    F_n = forward(seq, emission, p, q, p0, len(seq))
    ll = np.logaddexp(F_n[0], F_n[1])
    for k in range(1, len(seq) + 1):
        F_k = forward(seq, emission, p, q, p0, k)
        B_k = backward(seq, emission, p, q, p0, k)
        log_posterior_for_each_state = (F_k + B_k) - ll # Because
        post.append(np.argmax(log_posterior_for_each_state) + 1) # Convert 0 to 1 and 1 to 2
    return ''.join(np.char.mod('%d', post))


def viterbi(seq: str, emission: pd.DataFrame, p:float, q: float, p0: float) -> str:
    seq = '^' + seq
    lnp = np.log(p)
    lnq = np.log(q)
    ln1p = np.log(1 - p)
    ln1q = np.log(1 - q)
    lne = emission.copy()
    for c in lne.columns:
        lne[c] = np.log(lne[c])

    V = np.zeros((2, len(seq)))
    Ptr = np.zeros((2, len(seq)), dtype=int)
    for i in range(1, len(seq)):
        for k in range(V.shape[0]):
            if i > 0:
                tau = [ln1p, lnq] if k == 0 else [lnp, ln1q]
            else:
                tau = [np.log(p0), np.log(1 - p0)]
            tau = np.array(tau)
            V[k,i] = lne.loc[k,seq[i]] + np.max(V[:,i-1] + tau)
            Ptr[k,i] = np.argmax(V[:,i-1] + tau)
    
    states = [np.argmax(V[:,-1])]
    for i in range(len(seq) - 1, 1, -1):
        states.append(Ptr[states[-1], i])
    return ''.join(str(s + 1) for s in states[::-1])



def print_hidden_states(seq_a: str, seq_b: str, chunk_size = 50) -> None:
    a_chunks = [seq_a[i:i+chunk_size] for i in range(0, len(seq_a), chunk_size)]
    b_chunks = [seq_b[i:i+chunk_size] for i in range(0, len(seq_b), chunk_size)]

    for ac, bc in zip(a_chunks, b_chunks):
        print(f"{ac}\n{bc}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq',
                        help='A sequence over the alphabet (e.g. ACTGGACTACGTCATGCA or 1621636142516152416616666166616)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emision.tsv)')
    parser.add_argument('p', help='probability to transition from state 1 to state 2 (e.g. 0.01)', type=float)
    parser.add_argument('q', help='probability to transition from state 2 to state 1 (e.g. 0.5)', type=float)
    parser.add_argument('p0', help='intial probability of entering state 1 (e.g. 0.9)', type=float)
    args = parser.parse_args()

    initial_emission = pd.read_table(args.initial_emission)

    if args.alg == 'viterbi':
        print_hidden_states(viterbi(args.seq, initial_emission, args.p, args.q, args.p0), args.seq)

    elif args.alg == 'forward':
        F_n = forward(args.seq, initial_emission, args.p, args.q, args.p0, len(args.seq))
        ll = np.logaddexp(F_n[0], F_n[1])
        print(ll)

    elif args.alg == 'backward':
        B_0 = backward(args.seq, initial_emission, args.p, args.q, args.p0, 0)
        ll = B_0[0]
        print(ll)

    elif args.alg == 'posterior':
        print_hidden_states(posterior(args.seq, initial_emission, args.p, args.q, args.p0), args.seq)


if __name__ == '__main__':
    main()
