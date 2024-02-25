import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import openai
import random
import json
from PIL import Image
import os

PATH_TO_FOLDER = '.'
PATH_TO_TEXT_FILES = PATH_TO_FOLDER + '/text-files'  # folder holding text files, typically GPT output
PATH_TO_SAVED_PLOTS = PATH_TO_FOLDER + '/plots'  # folder holding plots, eg, network figures
PATH_TO_STATS_FILES = PATH_TO_FOLDER + '/stats'  # folder holding stats files, eg, proportion of nodes in giant component
DEFAULT_TEMPERATURE = 0.8
SHOW_PLOTS = False
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

def draw_and_save_network_plot(G, save_prefix):
    """
    Draw network, save figure.
    """
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0, k=2*1/np.sqrt(len(G.nodes()))))
    plt.axis("off")  # turn off axis
    axis = plt.gca()
    axis.set_xlim([1.1*x for x in axis.get_xlim()])  # add padding so that node labels aren't cut off
    axis.set_ylim([1.1*y for y in axis.get_ylim()])
    plt.tight_layout()
    fig_path = os.path.join(PATH_TO_SAVED_PLOTS, f'{save_prefix}.png')
    print('Saving network drawing in ', fig_path)
    plt.savefig(fig_path)

def save_network(G, save_prefix):
    """
    Save network as adjlist.
    """
    graph_path = os.path.join(PATH_TO_TEXT_FILES, f'{save_prefix}.adj')
    print('Saving adjlist in ', graph_path)
    nx.write_adjlist(G, graph_path)

def extract_gpt_output(response, savename=None):
    """
    Extract output message from GPT, check for finish reason.
    """

    if savename is not None:
        # read json in savename
        # if file exists
        if os.path.exists(savename):
            with open(savename) as f:
                data = json.load(f)
        else:
            data = {"prompt_tokens": 0, "completion_tokens": 0}

        data["prompt_tokens"] += response.usage.prompt_tokens
        data["completion_tokens"] += response.usage.completion_tokens


        # save to savename
        with open(savename, 'w') as f:
            json.dump(data, f)

    response = response.choices[0]
    finish_reason = response.finish_reason
    if finish_reason != 'stop':
        raise Exception(f'Response stopped for reason {finish_reason}')
    return response.message.content

def get_node_from_string(s):
    """
    If it is a persona of the form "<name> - <description>", get name; else, assume to be name.
    Replace spaces in name with hyphens, so that we can save to and read from nx adjlist.
    """
    if ' - ' in s:  # seems to be persona
        s = s.split(' - ', 1)[0]
    node = s.replace(' ', '-')
    return node

def prop_nodes_in_giant_component(G):
    """
    Get proportion of nodes in largest conneced component.
    """
    largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
    return len(largest_cc) / len(G.nodes())

def shuffle_dict(dict):
    temp = list(dict.keys())
    random.shuffle(temp)
    shuffled_dict = {}
    for item in temp:
        shuffled_dict[item] = dict[item]
        
    return shuffled_dict

def compute_token_cost(savepath, nr_networks, model='gpt-3.5-turbo'):

    prompt_tokens = []
    completion_tokens = []
    for i in range(nr_networks):
        with open(f'{savepath}-{i}.json') as f:
            data = json.load(f)
            prompt_tokens.append(data['prompt_tokens'])
            completion_tokens.append(data['completion_tokens'])

    # print averages and std
    print(f'Files in {savepath}: {nr_networks}')
    print(f'Prompt tokens: {np.mean(prompt_tokens)} +- {np.std(prompt_tokens)}')
    print(f'Completion tokens: {np.mean(completion_tokens)} +- {np.std(completion_tokens)}')

    # pricing
    if model == 'gpt-3.5-turbo':
        prompt_cost = 0.0005/1000
        completion_cost = 0.0015/1000
        costs = [prompt_cost*pt + completion_cost*ct for pt, ct in zip(prompt_tokens, completion_tokens)]
        print(f'Cost in dollars: {np.mean(costs)} +- {np.std(costs)}')

    else:
        print("Model cost unknown")

def combine_plots(folders, plot_names):
    for j, plot_name in enumerate(plot_names):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        for i, folder in enumerate(folders):
            img_path = os.path.join(folder, plot_name)
            img = Image.open(img_path)
            pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
            axs[pairs[i]].imshow(img)
            axs[pairs[i]].axis('off')

        plt.tight_layout()
        # save combined plot
        fig_path = os.path.join(os.path.join(PATH_TO_SAVED_PLOTS), f'{plot_name}_combined_plot.png')
        # save plot
        print('Saving combined plot in ', fig_path)
        plt.savefig(fig_path)


if __name__ == '__main__':

    compute_token_cost('costs/cost_all-at-once-for_us_50-gpt-3.5-turbo', 15)
    compute_token_cost('costs/cost_llm-as-agent-for_us_50-gpt-3.5-turbo', 15)
    compute_token_cost('costs/cost_one-by-one-for_us_50-gpt-3.5-turbo', 15)

    combine_plots(['plots/all-at-once-for_us_50-gpt-3.5-turbo', 'plots/llm-as-agent-for_us_50-gpt-3.5-turbo', 'plots/one-by-one-for_us_50-gpt-3.5-turbo', 'plots/real'],
                  ['betweenness_centrality_hist.png', 'degree_centrality_hist.png', 'closeness_centrality_hist.png'])
