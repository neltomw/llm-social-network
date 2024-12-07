from collections import Counter 
import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import seaborn as sns
from scipy.stats import ks_2samp
import time

import plotting
from constants_and_utils import *
from generate_personas import *

def load_list_of_graphs(prefix, start_seed, end_seed, directed=True, include_ts=False):
    """
    Load list of graphs from adjlist. By default, assume directed graphs.
    """
    list_of_G = []
    min_time, max_time = None, None
    for s in range(start_seed, end_seed):
        fn = os.path.join(PATH_TO_TEXT_FILES, f'{prefix}_{s}.adj')
        if os.path.exists(fn):
            mod = time.ctime(os.path.getmtime(fn))  # last modified time
            if (min_time is None) or (mod < min_time):
                min_time = mod 
            elif (max_time is None) or (mod > max_time):
                max_time = mod 
            if directed:
                G = nx.read_adjlist(fn, create_using=nx.DiGraph)
            else:
                G = nx.read_adjlist(fn)
            list_of_G.append(G)
    if include_ts:
        return list_of_G, min_time, max_time
    return list_of_G

def get_edge_proportions(list_of_G):
    """
    What proportion of the time does each edge appear?
    """
    edge_counts = {}
    # initialize all possible edge counts to 0
    nodes = list_of_G[0].nodes()
    for ni in nodes:
        for nj in nodes:
#            if ni != nj:
            edge_counts[(ni, nj)] = 0
    assert len(edge_counts) == (len(nodes) * (len(nodes))) # CHANGE
    # add actual edges
    for G in list_of_G:
        for e in G.edges():
            edge_counts[e] = edge_counts[e] + 1
    # sort by highest to lowest count
    sorted_edges = sorted(edge_counts.keys(), key=lambda x: -edge_counts[x])
    sorted_props = [edge_counts[e]/len(list_of_G) for e in sorted_edges]
    return sorted_edges, sorted_props

def compute_edge_distance(G1, G2):
    """
    Out of all possible edges, how often do G1 and G2 disagree? 
    Disagree means edge is present in one and missing in the other.
    Return the proportion of edges where G1 and G2 disagree.
    """
    assert set(G1.nodes()) == set(G2.nodes())
    E1 = set(G1.edges())
    E2 = set(G2.edges())
    distance = len(E1 - E2)  # present in G1 but missing in G2
    distance += len(E2 - E1)  # present in G2 but missing in G1
    num_nodes = len(G1.nodes())
    num_edges = num_nodes * (num_nodes-1)  # total num possible edges
    return distance / num_edges

def get_edge_summary(list_of_G, save_name):
    """
    Summarize edge-related statistics: 
    1) num edges per graph,
    2) average edge distance between graph pairs,
    3) proportion of graphs that each edge appears.
    """
    num_edges = [len(G.edges()) for G in list_of_G]

    plotting.plot_edges(num_edges, save_name)

    all_real_d = []
    for i, G1 in enumerate(list_of_G):
        if i < (len(list_of_G)-1):
            for G2 in list_of_G[i+1:]:
                all_real_d.append(compute_edge_distance(G1, G2))
    print('Average edge distance between graphs: %.3f' % np.mean(all_real_d))

    plotting.plot_edge_dist(all_real_d, save_name)

    edges, props = get_edge_proportions(list_of_G)
    print('Most common edges:')
    for i in range(30):
        print('%d. %s -> %s (p=%.3f)' % (i, edges[i][0], edges[i][1], props[i]))

    plotting.plot_props(props, edges, save_name)

def compute_exp_cross_from_group_counts(group_counts):
    """
    Compute expected proportion of edges that are cross-relations, given
    number of nodes that are in each group.
    This matches the method below that uses the complete graph.
    """
    groups = list(group_counts.keys())
    cr_total = 0
    for i1, g1 in enumerate(groups[:-1]):
        for g2 in groups[i1+1:]:
            cr_total += group_counts[g1] * group_counts[g2]
    num_nodes = np.sum(list(group_counts.values()))
    total_num_edges = num_nodes * (num_nodes-1) / 2
    return cr_total / total_num_edges 

def compute_cross_proportions(G, personas, demo_keys, ratio=True):
    """
    Compute proportion of edges that are cross-relations, per demographic variable.
    If ratio is true, divide by expected proportions.
    """
    observed = _compute_cross_proportions(G, personas, demo_keys)
    if not ratio:
        return observed 
    complete = nx.complete_graph(G.nodes())
    expected = _compute_cross_proportions(complete, personas, demo_keys)
    return observed / expected

def _compute_cross_proportions(G, personas, demo_keys):
    """
    Helper function to compute the proportion of edges in the graph that are 
    cross-relations, per demographic variable.
    """ 
    # count cross-relationships in graph
    crs = np.zeros(len(demo_keys))
    for source, target in G.edges():
        demo1 = personas[source]
        demo2 = personas[target]
        for ind, d in enumerate(demo_keys):
            if d == 'age':  # take absolute difference for age
                diff = abs(int(demo1[d]) - int(demo2[d]))
            else:
                diff = int(demo1[d] != demo2[d])  # 1 if they are different, 0 otherwise
            crs[ind] += diff
    # get proportion of edges that are cross-relations or average difference in age
    props = crs / len(G.edges())  
    return props

def compute_same_proportions(G, personas, demo_keys, ratio=True):
    """
    Compute proportion of edges that are same-group relations, per demographic variable.
    If ratio is true, divide by expected proportions.
    """
    observed = _compute_same_proportions(G, personas, demo_keys, True)
    print("observed", observed)
    if not ratio:
        return observed 
    complete = nx.complete_graph(G.nodes())
    print("complete", complete)
    expected = _compute_same_proportions(complete, personas, demo_keys)
    print("expected", expected)
    return observed / expected

def _compute_same_proportions(G, personas, demo_keys, should_print=False):
    """
    Helper function to compute the proportion of edges in the graph that are 
    same relations, per demographic variable.
    """ 
    # count same-relationships in graph

    political_connections = {}

    same_counts = np.zeros(len(demo_keys))
    for source, target in G.edges():
        demo1 = personas[source]
        demo2 = personas[target]
        if should_print:
            print("source", source)
            print("target", target)
            print("demo1", demo1)
            print("demo2", demo2)
        for ind, d in enumerate(demo_keys):
            if d == 'political affiliation':
                #print('political affiliation', int(demo1[d] == demo2[d]), demo1[d], demo2[d])
                if demo1[d] +'-'+ demo2[d] not in political_connections:
                    political_connections[demo1[d] +'-'+ demo2[d]] = 0
                political_connections[demo1[d] +'-'+ demo2[d]] = political_connections[demo1[d] +'-'+ demo2[d]] + 1 / len(G.edges())
            if d == 'age':  # check whether age is within 10
                same = int(abs(int(demo1[d]) - int(demo2[d])) <= 10)
            else:
                same = int(demo1[d] == demo2[d])
            same_counts[ind] += same
    # get proportion of edges that are same relation
    props = same_counts / len(G.edges())
    print("political_connections", political_connections)
    print("len(G.edges())", len(G.edges()))
    return props

def summarize_network_metrics(list_of_G, personas, demo_keys, save_name, demos=True):

    if not os.path.exists(os.path.join(PATH_TO_STATS_FILES, f'{save_name}')):
        os.makedirs(os.path.join(PATH_TO_STATS_FILES, f'{save_name}'))

    ### ---------------------------------- homophily ---------------------------------- ###
    if demos:
        homophily_metrics_df = pd.DataFrame({'graph_nr':[], 'demo':[], '_metric_value':[], 'save_name':[]})
        print("homophily_metrics_df", homophily_metrics_df)
        for graph_nr, G in enumerate(list_of_G):
            same_homophily = list(compute_same_proportions(G, personas, demo_keys, ratio=True))
            print("same_homophily", same_homophily)
            same_df = pd.DataFrame({'graph_nr':graph_nr, 'demo':demo_keys, 'metric_name': 'same_ratio',
                                    '_metric_value':same_homophily, 'save_name':[save_name]*len(demo_keys)})
            print("same_df", same_df)
            cross_homophily = list(compute_cross_proportions(G, personas, demo_keys, ratio=True))
            print("cross_homophily", cross_homophily)
            cross_df = pd.DataFrame({'graph_nr':graph_nr, 'demo':demo_keys, 'metric_name': 'cross_ratio',
                                    '_metric_value':cross_homophily, 'save_name':[save_name]*len(demo_keys)})
            print("cross_df", cross_df)
            # concat with series
            homophily_metrics_df = pd.concat([homophily_metrics_df, same_df, cross_df])
            print("homophily_metrics_df", homophily_metrics_df)
        # save homophily metrics dataframe in stats
        fn = f'{save_name}/homophily.csv'
        homophily_metrics_df.to_csv(os.path.join(PATH_TO_STATS_FILES, fn), index=False)
        print('Saved homophily metrics to ' + fn)

    ### ---------------------------------- scalar network metrics ---------------------------------- ###
    network_metrics_df = pd.DataFrame({'graph_nr':[], 'metric_name':[], '_metric_value':[], 'save_name':[]})

    network_metrics = ['density', 'avg_clustering_coef', 'prop_nodes_lcc', 'radius', 'diameter', 'avg_shortest_path', 'modularity']  # 'assortativity', 
    network_func = [nx.density, nx.average_clustering, prop_nodes_in_giant_component, nx.radius, nx.diameter, nx.average_shortest_path_length, nx.community.modularity]
    for graph_nr, G in enumerate(list_of_G):

        for metric_name, f in zip(network_metrics, network_func):
            if metric_name in ['radius', 'diameter', 'avg_shortest_path']:
                # use LCC for connectivity measures
                largest_cc = sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0]
                _metric_value = f(G.subgraph(largest_cc).to_undirected()) / max(1, np.log(len(largest_cc)))
            elif metric_name == 'modularity':
                comms = nx.community.louvain_communities(G.to_undirected())  # get communities with Louvain
                _metric_value = f(G.to_undirected(), comms)
            else:
                _metric_value = f(G.to_undirected())

            network_metrics_df = pd.concat([network_metrics_df, pd.DataFrame({'graph_nr':graph_nr,
                                                                              'metric_name':[metric_name],
                                                                              '_metric_value':[_metric_value],
                                                                              'save_name':[save_name]})])

    ### ---------------------------------- node-level network metrics ---------------------------------- ###
    node_metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
    node_func = [nx.degree_centrality, nx.betweenness_centrality, nx.closeness_centrality]
    for graph_nr, G in enumerate(list_of_G):
        for metric_name, f in zip(node_metrics, node_func):
            metric_dict = f(G.to_undirected())
            temp_df = pd.DataFrame(metric_dict.items(), columns=['node', '_metric_value'])
            temp_df['graph_nr'] = graph_nr
            temp_df['metric_name'] = metric_name
            temp_df['save_name'] = save_name
            network_metrics_df = pd.concat([network_metrics_df, temp_df])

    # save network metrics dataframe in stats
    fn = f'{save_name}/network_metrics.csv'
    network_metrics_df.to_csv(os.path.join(PATH_TO_STATS_FILES, fn), index=False)
    print("Saved network metrics to " + fn)


def compute_pairwise_ratios(G, personas, demo, cutoff=1):
    """
    Compute matrix where m[a, b] represents the observed number of edges between
    nodes in group a and b divided by expected number of such edges.
    """
    vals = [personas[k][demo] for k in personas]
    groups = [g for g,c in Counter(vals).most_common() if c >= cutoff]
    obs_mat = _compute_pairwise_props(G, personas, demo, groups)
    exp_mat = _compute_pairwise_props(nx.complete_graph(G.nodes()), personas, demo, groups)
    ratio = obs_mat / exp_mat
    return groups, ratio


def _compute_pairwise_props(G, personas, demo, groups):
    assert type(groups) == list 
    mat = np.zeros((len(groups), len(groups)))
    for u,v in G.to_undirected().edges():
        if (personas[u][demo] in groups) and (personas[v][demo] in groups):
            g1 = groups.index(personas[u][demo])
            g2 = groups.index(personas[v][demo])
            mat[g1][g2] += 1
            mat[g2][g1] += 1
    return mat / np.sum(mat)
    

def compute_isolation_index(G, personas):
    """
    Compute political isolation index, following Halberstam and Knight (2016).
    """
    nodes = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodes).todense()
    politics = np.array([personas[n]['political affiliation'] for n in nodes])
    
    # Convert to numpy arrays to ensure proper operations
    A = np.array(A)
    
    # compute share conservative
    num_neighbors_c = A @ (politics == 'Republican').astype(int)
    num_neighbors_l = A @ (politics == 'Democrat').astype(int)

    #print("Computing share_conservative...")
    denominators = (num_neighbors_c + num_neighbors_l)
    share_conservative = np.divide(num_neighbors_c, denominators, 
                                 out=np.zeros_like(num_neighbors_c, dtype=float),
                                 where=denominators!=0)
    #print("share_conservative computed:", share_conservative[:5])  # Print first 5 values
    
    #print("Computing degree...")
    degree = np.sum(A, axis=1)  # Changed from axis=0 to axis=1 for row sums
    #print("degree computed:", degree[:5])  # Print first 5 values
    
    #print("Computing conservative_exposure...")
    conservative_exposure = np.divide(A @ share_conservative, degree,
                                   out=np.zeros_like(degree, dtype=float),
                                   where=degree!=0)
    #print("conservative_exposure computed:", conservative_exposure[:5])
    
    # compute isolation
    avg_exposure_c = np.mean(conservative_exposure[politics == 'Republican'])
    avg_exposure_l = np.mean(conservative_exposure[politics == 'Democrat'])
    isolation = avg_exposure_c - avg_exposure_l
    
    #print("Final metrics computed")
    #print(f"avg_exposure_c: {avg_exposure_c}")
    #print(f"avg_exposure_l: {avg_exposure_l}")
    #print(f"isolation: {isolation}")
    
    return isolation, avg_exposure_c, avg_exposure_l

def compute_polarization(G, personas):
    """
    Compute polarization, following Garimella and Weber (2017).
    """
    nodes = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodes).todense()
    politics = np.array([personas[n]['political affiliation'] for n in nodes])
    assert A.shape == (len(politics), len(politics))

    alpha = np.ones(len(politics))
    beta = np.ones(len(politics))
    alpha += A @ (politics == 'Democrat').astype(int)
    beta += A @ (politics == 'Republican').astype(int)
    lean = alpha / (alpha + beta)
    pol = 2 * np.abs(0.5 - lean)
    return pol 

def plot_expected_vs_observed_age_gaps(list_of_G, personas):
    """
    """
    obs_gaps = []
    for G in list_of_G:
        for (u,v) in G.edges():
            gap = np.abs(personas[u]['age'] - personas[v]['age'])
            obs_gaps.append(gap)
    
    exp_gaps = []
    complete = nx.complete_graph(list_of_G[0].nodes())
    for (u,v) in complete.edges():
        gap = np.abs(personas[u]['age'] - personas[v]['age'])
        exp_gaps.append(gap)

    bins = np.arange(0, 101, 5)
    plt.figure(figsize=(6,4))
    plt.hist(exp_gaps, color='tab:blue', label='expected', density=True, bins=bins)
    plt.hist(obs_gaps, color='tab:orange', alpha=0.5, density=True, label='observed', bins=bins)
    plt.xlabel('Age gap btwn friends', fontsize=16)
    plt.grid(alpha=0.2)
    # ymin, ymax = plt.ylim()
    # plt.vlines([np.mean(exp_gaps)], ymin, ymax, color='tab:blue', label=f'exp mean={np.mean(exp_gaps):0.3f}')
    # plt.vlines([np.mean(obs_gaps)], ymin, ymax, color='tab:orange', label=f'obs mean={np.mean(obs_gaps):0.3f}')
    plt.legend()

def compare_network_metrics(network_df, metric_name, save_name):
    """
    Get quantitative comparison of real and generated networks.
    """
    real_vals = network_df[(network_df.save_name == 'real') & (network_df.metric_name == metric_name)]['_metric_value'].values
    gen_vals = network_df[(network_df.save_name == save_name) & (network_df.metric_name == metric_name)]['_metric_value'].values
    print(f'Found {len(real_vals)} real and {len(gen_vals)} generated values')
    mean_diff = np.abs(np.mean(real_vals)-np.mean(gen_vals))
    mean_diff_norm = mean_diff / np.std(real_vals)
    res = ks_2samp(real_vals, gen_vals)
    return mean_diff, mean_diff_norm, res.statistic, res.pvalue

def parse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    # Add arguments
    parser.add_argument('--persona_fn', type=str, default='us_50_with_names_with_interests', help='What is the name of the persona file you want to use?')
    parser.add_argument('--network_fn', type=str, help='What is the name of the network file you want to use?')
    parser.add_argument('--num_networks', type=int, help='How many networks are there?')
    parser.add_argument('--network_name', type=str, default='')
    parser.add_argument('--demos_to_include', nargs='+', default=['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation'])

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Persona file", args.persona_fn)
    print("Network file", args.network_fn)
    print("Number of networks", args.num_networks)
    
    return args


def count_communities(list_of_G, save_name):

    counts = []
    sizes = []
    mods = []
    for G in list_of_G:
        comms = nx.community.louvain_communities(G, seed=42)
        counts.append(len(comms))
        sizes = sizes + [len(c) for c in comms]

        modularity = nx.community.modularity(G, comms)
        mods.append(modularity)

    plotting.plot_communities(counts, sizes, mods, save_name)


if __name__ == '__main__':

    args = parse()
    list_of_G = load_list_of_graphs(args.network_fn, 0, args.num_networks)
    get_edge_summary(list_of_G, args.network_fn)
    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn) + '_' + args.network_name + '.json'
    # load from json
    with open(fn, 'r') as f:
        personas = json.load(f)

    summarize_network_metrics(list_of_G, personas, args.demos_to_include, save_name=args.network_fn)

    # python analyze_networks.py --persona_fn us_50_with_names_with_interests.json --network_fn llm-as-agent-us-50-gpt-3.5-turbo --num_networks 10
    # python analyze_networks.py --persona_fn us_50_with_names_with_interests.json --network_fn llm-as-agent-us-50-gpt-3.5-turbo --num_networks 10