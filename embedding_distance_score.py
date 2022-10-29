from pynauty import autgrp, Graph
from networkx.convert import to_edgelist
from numpy import array, matmul, average, sum as numpysum
from numpy.linalg import inv, norm, eig

# turn an networkx graph into a pynauty graph
# to generate the automorphism group elements
def nx_to_pn(nx_graph):
    n_nodes=nx_graph.number_of_nodes()
    pn_graph=Graph(n_nodes)
    for u,v in graph_to_edgelist(nx_graph):
        pn_graph.connect_vertex(u,[v])
        
    return pn_graph

def graph_to_edgelist(g):
    return [e[:2] for e in to_edgelist(g)]

# determine whether 2 different graph layouts are identical / similar (ie up to automorphisms and affine transformations)
def embedding_distance_score(nx_graph, pos1, pos2):

    # draw(nx_graph, pos=pos1)
    # draw_networkx_labels(nx_graph, pos=pos1)

    pn_graph=nx_to_pn(nx_graph)
    generators, grpsize1, grpsize2, orbits, numorbits=autgrp(pn_graph)
    
    # print(f"graph edgelist: {graph_to_edgelist(nx_graph)}\ngenerators: {generators}\ngrpsize1: {grpsize1}\ngrpsize2: {grpsize2}\norbits: {orbits}\nnumorbits: {numorbits}")

    # generate automorphism group
    automorphisms=get_automorphisms_from_generators(nx_graph, generators, int(grpsize1))

    
    # check if embeddings are identical
    # return score if necessary
    min_score=99999
    


    for automorphism in automorphisms:
        pos1_post_permutation = apply_transformation_to_pos(automorphism, pos1)
        score = affine_transformation_distance(pos1_post_permutation, pos2)
        min_score=min(min_score, score)
    return min_score


# Determine whether pos2 is an affine transformation of pos1
# [AA]               [1]           [BB]
# [AA]               [.]           [BB]
# [yy]            +  [.]         = [zz]
# [yy]               [.]           [zz]
# [..] [x x]         [.]           [..]
# [yy] [x x]         [1][t t]      [zz]

# pos1 (transform) + (translate) = pos2


# first normalise the pos matrices
# to be 0 centred, with avg magnitude 1
# then determine if 1 is a linear transformation of the other


# Overdetermined system, so trivial to solve
# Restrict to the top 2 rows, solve for x, then check to see if the rest works
# We will consider the magnitude of the average euclidean distance
def affine_transformation_distance(pos1, pos2):

    

    Ay=pos_to_normalised_matrix(pos1)
    Bz=pos_to_normalised_matrix(pos2)

    # assert find_best_linear_transformation_distance_parallelised(Ay, Bz)==find_best_linear_transformation_distance(Ay, Bz)
    return find_best_linear_transformation_distance_parallelised(Ay, Bz)


# given a pos dictionary, give us a coordinate array
def pos_to_matrix(pos):
    l=[None]*len(pos)
    for i in pos:
        l[i]=pos[i]
    return array(l)


# overdetermined system
# iterate through pairs of rows
# work out which one yields best transformation
def find_best_linear_transformation_distance_parallelised(m1, m2):
    height=m1.shape[0]
    As=[]
    Bs=[]
    for i in range(height):
        for j in range(i+1, height):
            A=m1[[i,j]] # take rows i and j and turn into a 2x2 matrix
            B=m2[[i,j]]
            As.append(A)
            Bs.append(B)
    
    count=len(As)
    As=array(As)
    Bs=array(Bs)
    InvAs=inv(As)
    xs=matmul(InvAs, Bs)
    m1xs=matmul(m1,xs)
    m2s=array([m2 for _ in range(count)])
    coordinate_wise_distances=m1xs-m2s
    pointwise_distances=norm(coordinate_wise_distances, axis=2)
    sum_distances=numpysum(pointwise_distances, axis=1)
    eigenvals, eigenvecs=eig(xs)
    eigenval_norms = norm(eigenvals, axis=1)
    transformation_distances=sum_distances*eigenval_norms
    best_distance=min(transformation_distances)
    return best_distance


# given pos, return coordinate array normalised to have 0 mean and 1 avg magnitude
# so that we can just worry about linear transformations instead of affine transformations
def pos_to_normalised_matrix(pos):
    unnormalised_matrix=pos_to_matrix(pos)
    means=average(unnormalised_matrix, axis=0)
    zero_centred_matrix=unnormalised_matrix-means
    magnitudes=norm(zero_centred_matrix, axis=1)
    avg_magnitude=average(magnitudes)
    normalised_matrix=zero_centred_matrix/avg_magnitude
    return normalised_matrix


# given a graph and the generators
# iteratively apply the generators until we have generated the group
def get_automorphisms_from_generators(graph, generators, target):

    # found is a set of tuples, initially only containing
    # the tuple corresponding to the identity automorphism
    found=set([tuple(sorted(graph.nodes))])
    prev_found_len=len(found)
    just_found=[]

    # main loop
    while True:

        # try applying every possible generator to every automorphism found
        for generator in generators:
            for element in found:
                new_element=tuple(apply_generator(element, generator))
                just_found.append(new_element)

        # if nothing new was found, break
        for j in just_found:
            found.add(j)
        just_found=[]

        # otherwise prepare for next loop
        if len(found)==prev_found_len:break
        else:prev_found_len=len(found)

    # make sure we have the correct number of automorphisms
    assert len(found)==target

    return list(found)


# apply automorphism transformation to pos dictionary 
def apply_transformation_to_pos(automorphism, pos):
    out={}
    # nth entry in pos should be mth entry in out
    for n,m in enumerate(automorphism):
        out[m]=pos[n]
    return out


# apply group generator to an element
def apply_generator(element, generator):
    out=[None]*len(element)
    for n,x in enumerate(generator):
        out[x]=element[n]
    return out