import torch

def combine_subgraphs(states):
    res = list(zip(*states))
    if len(res) == 2:
        xs, eis = res
        gvs = None
    elif len(res) == 3:
        xs, eis, gvs = res

    # ei we need to update each node idx to be
    # ei[i] += len(ei[i-1])
    offset = 0
    new_eis = []
    for i in range(len(eis)):
        new_eis.append(eis[i] + offset)
        offset += xs[i].size(0)

    # X is easy, just cat
    xs = torch.cat(xs, dim=0)
    eis = torch.cat(new_eis, dim=1)
    
    if gvs is not None:
        gvs = torch.cat(gvs, dim=0)
        return xs, eis, gvs
    return xs, eis