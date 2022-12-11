import argparse
import numpy as np
import pandas as pd

def forward(seq: str, emission: pd.DataFrame, p:float, q: float, p0: float) -> float:
    table = np.zeros((emission.shape[0], len(seq))) # A row for each state
    table[0,0] = np.log(p0 * emission.loc[0, seq[0]])
    table[1,0] = np.log((1 - p0)  * emission.loc[1, seq[0]])

    lnp = np.log(p)
    lnq = np.log(q)
    ln1p = np.log(1 - p)
    ln1q = np.log(1 - q)
    lne = emission.copy()
    for c in lne.columns:
        lne[c] = np.log(lne[c])

    for i in range(1, len(seq)):
        for j in range(table.shape[0]):
            tau = [ln1p, lnq] if j == 0 else [lnp, ln1q]
            tau = np.array(tau)

            ln_vals_to_sum = table[:,i-1] + tau + lne.loc[j, seq[i]]
            table[j,i] = ln_vals_to_sum[0] + np.log(1 + np.exp(ln_vals_to_sum[1] - ln_vals_to_sum[0]))
    
    return table[0, -1] + np.log(1 + np.exp(table[1, -1] - table[0, -1]))


def backward(seq: str, emission: pd.DataFrame, p:float, q: float, p0: float) -> float:
    seq = '^' + seq
    table = np.zeros((emission.shape[0], len(seq))) # A row for each state
    lnp = np.log(p)
    lnq = np.log(q)
    ln1p = np.log(1 - p)
    ln1q = np.log(1 - q)
    lne = emission.copy()
    for c in lne.columns:
        lne[c] = np.log(lne[c])

    for i in range(len(seq) - 2, -1, -1):
        for j in range(table.shape[0]):
            if i > 0:
                tau = [ln1p, lnp] if j == 0 else [lnq, ln1q]
            else:
                tau = [np.log(p0), np.log(1 - p0)]
            tau = np.array(tau)

            ln_vals_to_sum = table[:,i+1] + tau + lne[seq[i + 1]].to_numpy()
            table[j,i] = np.logaddexp(ln_vals_to_sum[0], ln_vals_to_sum[1])
    return table[0, 0]


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
        raise NotImplementedError

    elif args.alg == 'forward':
        print(f'{forward(args.seq, initial_emission, args.p, args.q, args.p0):.2f}')

    elif args.alg == 'backward':
        print(f'{backward(args.seq, initial_emission, args.p, args.q, args.p0):.2f}')

    elif args.alg == 'posterior':
        raise NotImplementedError


if __name__ == '__main__':
    main()
