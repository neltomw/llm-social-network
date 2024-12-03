from constants_and_utils import *
from generate_personas import *
import argparse
import json
import numpy as np
import pandas as pd
import time
import multiprocessing as mp
from functools import partial

def generate_single_network(seed, args, personas, pids, save_prefix, demos_to_include):
    """
    Generate a single network with the given seed. This function will be run in parallel.
    """
    ts = time.time()
    np.random.seed(seed)
    order = np.random.choice(pids, size=len(pids), replace=False)
    print(f'Seed {seed} - Order of printing:', order[:10])
    
    G, reasons, num_tries, input_toks, output_toks = generate_network(
        args.method, demos_to_include, personas, order, args.model, 
        mean_choices=args.mean_choices if args.mean_choices > 0 else None,
        include_reason=args.include_reason, all_demos=args.prompt_all, 
        only_degree=not args.include_friend_list, temp=args.temp, 
        num_iter=args.num_iter, verbose=args.verbose)
    
    # Save network and plot
    save_network(G, f'{save_prefix}_{seed}')
    draw_and_save_network_plot(G, f'{save_prefix}_{seed}')
    
    duration = time.time() - ts
    print(f'Seed {seed}: {len(G.edges())} edges, num tries={num_tries}, input toks={input_toks}, output toks={output_toks} [time={duration:.2f}s]')
    
    # Save reasons if needed
    if args.include_reason:
        fn = os.path.join(PATH_TO_TEXT_FILES, f'{save_prefix}_{seed}_reasons.json')
        with open(fn, 'w') as f:
            json.dump(reasons, f)
    
    return {
        'seed': seed,
        'duration': duration,
        'num_tries': num_tries,
        'num_input_toks': input_toks,
        'num_output_toks': output_toks
    }

from constants_and_utils import *
from generate_personas import *
import argparse
import json
import numpy as np
import pandas as pd
import time

def get_persona_format(demos_to_include):
    """
    Define persona format for GPT: eg, "ID. Name - Gender, Age, Race/ethnicity, Religion, Political Affiliation". 
    """
    persona_format = 'ID. '
    if 'name' in demos_to_include:
        persona_format += 'Name - '
    for demo in demos_to_include:
        if demo != 'name':
            persona_format += f'{demo.capitalize()}, '
    persona_format = persona_format[:-2]  # remove trailing ', '
    return persona_format


def get_system_prompt(method, personas, demos_to_include, curr_pid=None, G=None, 
                      only_degree=True, num_choices=None, include_reason=False, all_demos=False):
    """
    Get content for system message.
    """
    assert method in {'global', 'global-expressive', 'local', 'sequential', 'iterative-add', 'iterative-drop'}
    if G is not None:
        assert 'iterative' in method 
    if (curr_pid is not None) or include_reason:
        assert method != 'global' or method != 'global-expressive'
    if num_choices is not None:
        assert method in {'local', 'sequential'}
        assert num_choices >= 1

    print("demos_to_include", demos_to_include)

    # commonly used strings
    persona_format = get_persona_format(demos_to_include)
    persona_format = f'where each person is described as \"{persona_format}\"'
    prompt_extra = 'Do not include any other text in your response. Do not include any people who are not listed below.'
    if all_demos:
        prompt_extra = 'Pay attention to all demographics. ' + prompt_extra
    if curr_pid is not None:
        prompt_personal = assign_persona_to_model(personas[curr_pid], demos_to_include) + '.'
    
    if method == 'global':
        prompt = 'Your task is to create a realistic social network. You will be provided a list of people in the network, ' + persona_format + '. Provide a list of friendship pairs in the format ID, ID with each pair separated by a newline. ' + prompt_extra

    elif method == 'global-expressive':
        if include_reason:
            #Each user should choose a wide array of connections, depending on their sociability [between 2 and 30] (very introverted people may only have 2 or 4, introverts might have 6, neutral people might have 10, extroverts might have 14, very extroverted people may have 20 connections or more).

            prompt = """
Your task is to create a realistic social network. You will be provided a list of people in the network, {persona_format}. Don't rely fully on demographic information or interests to make connections, feel free to make things up. Here are 150 examples of potential types of relationships you could make up between the people in the network:

Family relationship (e.g., parent, sibling, cousin)
Romantic relationship (e.g., spouse, partner)
Is the parent of
Is the child of
Is the sibling of
Is the cousin of
Is married to
Is divorced from
Is the grandparent of
Is the grandchild of
Is the step-parent of
Is the stepchild of
Is the aunt/uncle of
Is the niece/nephew of
Is the in-law of
Is the godparent of
Is the godchild of
Is the foster parent of
Is the foster child of
Is the half-sibling of
Is engaged to
Is the ex-partner of
Met at a family reunion
Grew up together in the same household
Was adopted by
Shares a distant ancestor with
Is the descendant of
Is the ancestor of
Is the legal guardian of
Was raised by
Is the twin of
Is the biological child of
Was separated at birth from
Is the child of a close friend of
Is in a blended family with
Is a cousin through marriage to
Is in a long-term relationship with
Is the boyfriend/girlfriend of
Met on a blind date with
Met on a dating app
Had a one-night stand with
Is romantically interested in
Had an unrequited love for
Was a high school sweetheart of
Was a college partner of
Broke up with
Is currently dating
Had a summer fling with
Was a secret lover of
Is in an open relationship with
Was engaged to
Met through a mutual friend
Is the co-worker of
Is the boss of
Is the subordinate of
Is the mentor of
Is the mentee of
Was hired by
Was fired by
Worked on a project together with
Shared an office with
Attended the same job training as
Was a competitor for the same job as
Took over the role of
Referred for a job by
Was laid off at the same time as
Shared an internship with
Was a classmate of
Was a college roommate of
Sat next to in class
Was a lab partner of
Was in the same graduating class as
Was a teacher of
Was a student of
Met during an extracurricular activity at school
Shared a study group with
Went on a school trip with
Met at a party
Was introduced by a mutual friend
Met at a coffee shop
Sat next to on a plane
Was a pen pal of
Shared a taxi ride with
Met during a vacation
Met in line for an event
Was a neighbor of
Ran into each other at a grocery store
Volunteered together at a charity
Organized an event with
Was a fellow member of a church group
Was in the same sports team as
Was a co-organizer of a community project
Miscellaneous
Met at a hospital
Shared the same Airbnb with
Connected on a social media platform
Was a fellow attendee of a concert
Was in the same book club as
Was a member of the same fan club
Participated in the same online challenge as
Was a fellow gamer in an online team
Took the same guided tour with
Had a chance meeting during a natural disaster evacuation
Co-worker at the same company
Classmate in the same school
Neighbor in the same residential area
Member of the same sports team
Member of the same religious group
Member of the same hobby club
Fellow volunteer in a charity organization
Friend from childhood
Shared attendance at the same event
Connected through a mutual friend
Business partner
Shared participation in an online forum
Collaborator on a research project
Fellow student in an online course
Fellow activist in a political movement
Shared membership in a professional organization
Alumni of the same university
Mentor-mentee relationship
Employer-employee relationship
Supplier-customer relationship
Competitor in the same industry
Attendee of the same workshop or seminar
Members of the same online gaming guild
Fellow attendees at a conference
Co-author of a publication
Connection through social media (e.g., Facebook, LinkedIn)
Buyer-seller connection in a marketplace
Trainer-trainee relationship
Roommates or housemates
Fellow travelers on the same trip
Shared participation in a fitness class
Fellow attendees of the same concert
Coach-player relationship
Shared subscription to the same book club
Connection through a dating app
Members of the same advocacy group
Fellow parents at a PTA meeting
Connection through a shared hobby (e.g., photography)
Part of the same creative team (e.g., film, theater)
Shared interest in a specific fandom
Followers of the same influencer
Shared lineage (e.g., ancestry website match)
Collaboration on a community project
Connection via mutual investment in a startup
Members of the same fraternity or sorority
Fellow attendees of a protest
Fellow players in the same e-sports team
Shared interest in a podcast or show=
Fellow pet owners in a pet meetup group
Attendees of the same religious pilgrimage
Co-owners of a shared property
Members of the same recovery group (e.g., AA)
Connection through a shared language exchange program
Shared attendance at a summer camp
Fellow bloggers or content creators in a niche
Fellow shareholders in a company
Connection through crowdfunding (e.g., Kickstarter backers)
Fellow competitors in a hackathon
Connection via matchmaking by a mutual acquaintance
Shared participation in a book-writing group
Fellow volunteers in disaster relief
Connection via a tutoring relationship
Members of the same virtual reality community
Shared attendance at a TED Talk
Connection through philanthropic donations
Members of the same fantasy sports league
Fellow team members in a corporate retreat
Shared residency in the same dormitory
Connection through a common hairstylist
Fellow patients in a group therapy session
Connection through the same personal trainer
Members of the same wine-tasting club
Fellow participants in a survival skills course
Connection through a sports fantasy league
Members of the same investment club
Connection via a live-streaming platform
Members of the same makerspace or fab lab
Fellow passengers on a long-term expedition
Shared enrollment in a language class
Connection through dog-walking groups
Fellow organizers of a fundraising event
Connection through a shared interest in genealogy
Fellow debaters in a debate club
Shared work in community gardening
Connection via hosting on the same platform (e.g., Airbnb)
Fellow members of a hackspace
Fellow participants in a cooking class
Connection via a shared personal chef
Members of the same stock market discussion group
Fellow members of a cosplay group
Connection through a mutual yoga instructor
Shared involvement in the same political campaign
Members of the same study-abroad program
Connection via shared public transport routes
Fellow gamers in a live-streaming community
Shared membership in a music band
Fellow advocates in a legal case
Connection through shared intellectual property rights
Members of the same neighborhood watch group
Connection through the same barbershop
Fellow gig workers on the same platform
Shared participation in a science fair
Connection through the same tour guide
Members of the same chess club
Fellow contributors to open-source projects
Fellow members of a trivia team
Shared activity in a bike-sharing group
Members of the same farmers’ market cooperative
Fellow members of a local theater troupe
Connection through the same babysitter
Shared attendance at a pop culture convention
Fellow subscribers of a niche magazine
Members of the same astronomy club
Shared interest in rare collectibles (e.g., stamps, coins)
Fellow carpool participants
Shared roles in a neighborhood clean-up initiative
Members of the same singing ensemble
Fellow writers in a scriptwriting circle
Connection through a shared music teacher
Members of the same carpentry workshop
Shared visits to the same travel destinations
Fellow members of an online trivia league
Shared investment in a co-op business
Members of the same pottery class
Connection through shared library memberships
Fellow painters in an art class
Connection through a local history group
Fellow attendees of a philosophy seminar
Members of the same hiking club
Fellow birdwatchers in the same reserve
Shared participation in the same fitness app challenges
Connection via shared attendance at a retreat
Fellow participants in a marathon training group
Connection via shared usage of a delivery service
Members of the same photography darkroom
Shared activities at a robotics workshop
Fellow innovators in a tech incubator
Shared advocacy for environmental causes
Members of the same urban farming group
Connection via mentorship in an incubator program
Fellow attendees at a stargazing event
Connection through local artisan crafts fairs
Members of the same improv comedy group
Fellow influencers in the same niche market
Connection through shared medical conditions
Shared interest in speed dating events
Members of the same cryptocurrency discussion group
Fellow hosts of a joint podcast

You will start by initially creating a number of groups of users who are all connected to each other. You decide how many groups to make and how many users to put in each group. The network should make sense. The format will be like this:

Don't add any extra markdown to your response, follow the format exactly as it is presented here.

GROUPS

Group: GROUP_REASON
Group Users Names: USERNAME1, USERNAME2, ..., USERNAMEX
Group User IDs: USERID1, USERID2, ..., USERIDX

Group: GROUP_REASON
Group Users Names: USERNAME1, USERNAME2, ..., USERNAMEX
Group User IDs: USERID1, USERID2, ..., USERIDX

Group: GROUP_REASON
Group Users Names: USERNAME1, USERNAME2, ..., USERNAMEX
Group User IDs: USERID1, USERID2, ..., USERIDX

For example

Group: Members of the same hiking club
Group Users Names: Elaine Thompson, Bill Franks, Martin Sullivan
Group User IDs: 19, 1, 21

Group: Friends from childhood
Group Users Names: Elaine Thompson, Bill Franks, Martin Sullivan, Ethan Campbell, Javier Morales
Group User IDs: 19, 1, 21, 45, 3

Group: Family members
Group Users Names: Elaine Thompson, Bill Franks, Martin Sullivan, Ethan Campbell, Javier Morales
Group User IDs: 19, 1, 21, 45, 3


AND you will create connections between individual users. The format will be like this:

For each and every user in the list, you will output text matching this format, but for that specific user, with their specific preferences accounted for: 

USERS

"User ID
NAME is SOCIABILITY_LEVEL. They have X connections.
NAME1 (user ID1) REASON
NAME2 (user ID2) REASON
NAME3 (user ID3) REASON
...
NAMEX (user IDX) REASON
The user connections are ID1, ID2, ID3, ..., IDX"

Each user should choose a wide array of connections, depending on their sociability, very introverted people will have a small number of connections, very extroverted people will have a large number of connections.

Specifics about this social graph: I want several tight-knit communities, I want the network graph to have a high clustering coefficient.

Then you will output a new line ad move on to the next user, until you have completed this task for all the users in the list.

Your output will end up looking something like this:

User 1  
Bill Franks is neutral. They have 4 connections.
Elaine Thompson (user 19) Members of the same hiking club
Martin Sullivan (user 21) Fellow birdwatchers in the same reserve
Ethan Campbell (user 45) Connection via mentorship in an incubator program
Javier Morales (user 3) Fellow contributors to open-source projects
The user connections are 19, 21, 45, 3

User 12  
Jenine Lewis is very extroverted. They have 9 connections
Lila Preston (user 33) Classmate in the same school
Tanner Blackhawk (user 42) Neighbor in the same residential area
Bill Franks (user 1) Mentor-mentee relationship
Eleanor Thompson (user 8) Fellow attendees at a conference
Patricia Thompson (user 41) Connected through a mutual friend
Carlos Herrera (user 22) Alumni of the same university
Miguel Sanchez (user 4) Competitor in the same industry
Carolyn Thompson (user 6) Business partner
Maria Torres (user 11) Member of the same sports team
The user connections are 33, 42, 1, 8, 41, 22, 4, 6, 11

User 37  
You are Frank Genovani. You are introverted. You have 2 connections. You are joining a social network. You are identifying your connections! Please choose from the list provided at the top. Feel free to make up connections that aren't explained by demographic information or interests, like family or friendship connections, or use connections that have already been identified for other users.
Arthur Hayes (user 7) Connected through a mutual friend
Eleanor Thompson (user 8) Romantic relationship (e.g., spouse, partner)
The user connections are 7, 8

...

Make sure you do this FOR EVERY SINGLE USER, no matter how long your response needs to be. YOU NEED TO DO THIS FOR EVERY USER, IN ORDER, UNTIL YOU GET TO THE BOTTOM OF THE LIST.
"""
        else:
            prompt = """
Your task is to create a realistic social network. You will be provided a list of people in the network, {persona_format}.
For each and every user in the list, you will output text matching this format, but for that specific user, with their specific preferences accounted for: 
"User ID
You are a GENDER, age AGE, RACE, RELIGION, POLITICS, interests include: INTEREST, INTEREST, INTEREST, INTEREST, .... You are joining a social network. You are identifying your connections! Please choose from the list provided at the top. Feel free to make up connections that aren't explained by demographic information or interests,  or use connections that have already been identified for other users.
You are connected with the users ID1, ID2, ID3, ID4, ID5, ..., IDX"

Each user should choose between 7 and 15 friends from the list.
Then you will output a new line ad move on to the next user, until you have completed this task for all the users in the list.

Your output will end up looking something like this:

User 1  
You are a Man, age 47, White, Protestant, Republican, interests include: Hunting, fishing, classic rock, church activities, patriotic events, home improvement. You are joining a social network. You are identifying your connections! Please choose from the list provided at the top. Feel free to make up connections that aren't explained by demographic information or interests,  or use connections that have already been identified for other users.
You are connected with the users 19, 21, 45, 3, 6

User 12  
You are a Man, age 34, White, Protestant, Republican, interests include: Finance, fitness, outdoor activities, history, technology, conservative politics. You are joining a social network. You are identifying your connections! Please choose from the list provided at the top. Feel free to make up connections that aren't explained by demographic information or interests,  or use connections that have already been identified for other users.
You are connected with the users 33, 42, 1, 8, 41, 22, 4

User 37  
You are a Woman, age 58, Asian, Catholic, Democrat, interests include: Volunteering, social justice, culinary arts, family activities, church community involvement. You are joining a social network. You are identifying your connections! Please choose from the list provided at the top. Feel free to make up connections that aren't explained by demographic information or interests,  or use connections that have already been identified for other users.
You are connected with the users 7, 11, 8, 40, 32

..."""
        prompt += prompt_extra
    
    elif method in {'local', 'sequential'}:
        prompt = prompt_personal + ' You are joining a social network.\n\nYou will be provided a list of people in the network, ' + persona_format
        if method == 'sequential':
            prompt += ', followed by '
            if only_degree:
                prompt += 'their current number of friends'
            else:
                prompt += 'their current friends\' IDs'
        prompt += '.\n\nWhich of these people will you become friends with? '
        if num_choices is not None:
            pp = 'people' if num_choices > 1 else 'person'
            prompt += f'Choose {num_choices} {pp}. '
        if include_reason:
            prompt += 'Provide a list of *YOUR* friends and a short reason for why you are befriending them, in the format:\nID, reason\nID, reason\n...\n\n'
        else:
            prompt += 'Provide a list of *YOUR* friends in the format ID, ID, ID, etc. ' 
        prompt += prompt_extra
    
    elif method == 'iterative-add':
        prompt = prompt_personal + ' You are part of a social network and you want to make a new friend.\n\nYou will be provided a list of potential new friends, ' + persona_format + ', followed by their total number of friends and number of mutual friends with you. '
        curr_friends = ', '.join(list(G.neighbors(curr_pid)))
        prompt += 'Keep in mind that you are already friends with IDs ' + curr_friends + '.\n\nWhich person in this list are you likeliest to befriend? '
        if include_reason:
            prompt += 'Provide your answer in JSON form: {\"new friend\": ID, \"reason\": reason for adding friend}. '
        else:
            prompt += 'Answer by providing ONLY this person\'s ID. '
        prompt += prompt_extra
    
    else:  # iterative-drop
        prompt = prompt_personal + ' Unfortunately, you are busy with work and unable to keep up all your friendships.\n\nYou will be provided a list of your current friends, ' + persona_format + ', followed by their total number of friends and number of mutual friends with you.'
        prompt += '\n\nWhich friend in this list are you likeliest to drop? '
        if include_reason:
            prompt += 'Provide your answer in JSON form: {\"dropped friend\": ID, \"reason\": reason for dropping friend}. '
        else:
            prompt += 'Answer by providing ONLY this friend\'s ID. '
        prompt += prompt_extra


    #print("system prompt", prompt)
    return prompt 


def get_user_prompt(method, personas, order, demos_to_include, curr_pid=None, 
                    G=None, only_degree=True):
    """
    Get content for user message.
    """
    assert method in {'global', 'global-expressive', 'local', 'sequential', 'iterative-add', 'iterative-drop'}        
    lines = []
    if method == 'global':
        for pid in order:
            lines.append(convert_persona_to_string(personas[pid], demos_to_include, pid=pid))

    elif method == 'global-expressive':
        for pid in order:
            lines.append(convert_persona_to_string(personas[pid], demos_to_include, pid=pid))
        lines.append("REMEMBER IT IS ABSOLUTELY VITAL THAT YOU PROCESS EVERY SINGLE USER, DON'T MISS ANY AND DON'T TRUNCATE THE RESULTS.")
    
    elif method == 'local':
        assert curr_pid is not None 
        for pid in order:
            if pid != curr_pid:
                lines.append(convert_persona_to_string(personas[pid], demos_to_include, pid=pid))
        assert len(lines) == (len(order)-1)
    
    elif method == 'sequential':
        assert curr_pid is not None 
        assert G is not None
        for pid in order:
            if pid != curr_pid:
                persona = convert_persona_to_string(personas[pid], demos_to_include, pid=pid)
                cand_friends = set(G.neighbors(pid))  # candidate's friends
                if only_degree:
                    persona += f'; has {len(cand_friends)} friends'
                else:
                    if len(cand_friends) == 0:
                        persona += '; no friends yet'
                    else:
                        persona += '; friends with IDs ' + ', '.join(cand_friends)
                lines.append(persona)
        assert len(lines) == (len(order)-1)
        
    else:  # iterative
        assert curr_pid is not None 
        assert G is not None
        friends = list(G.neighbors(curr_pid))
        if method == 'iterative-add':
            id_list = list(set(G.nodes()) - set(friends) - {curr_pid})  # non-friends
            action = 'befriend'
        else:
            id_list = friends  # current friends
            action = 'drop'
        random.shuffle(id_list)
        for pid in id_list:
            persona = convert_persona_to_string(personas[pid], demos_to_include, pid=pid)
            cand_friends = set(G.neighbors(pid))  # candidate's friends
            mutuals = set(friends).intersection(cand_friends)
            lines.append(persona + f'; # friends: {len(cand_friends)}, # mutual friends: {len(mutuals)}')
        id_list = ', '.join(id_list)
        lines.append(f'Which person ID out of {id_list} are you likeliest to {action}?')
    
    prompt = '\n'.join(lines)

    #print("user prompt", prompt)
    return prompt 
    

def update_graph_from_response(method, response, G, curr_pid=None, include_reason=False, num_choices=None):
    """
    Parse response from LLM and update graph based on edges found.
    Expectation:
    - 'global' response should list all edges in the graph
    - 'global-expressive' response should list all edges in the graph
    - 'local' and 'sequential' should list all new edges for curr_pid
    - 'iterative-add' should list one new edge to add for curr_pid
    - 'iterative-drop' should list one existing edge to drop for curr_pid
    """
    assert method in {'global', 'global-expressive', 'local', 'sequential', 'iterative-add', 'iterative-drop'}
    if num_choices is not None:
        assert method in {'local', 'sequential'}
    if include_reason:
        assert method != 'global' and curr_pid is not None
        reasons = {}
    edges_found = []
    
    lines = response.split('\n')
    if method == 'global':
        for line in lines:
            id1, id2 = line.split(',')
            edges_found.append((id1.strip(), id2.strip()))
    
    elif method == 'global-expressive':
        user_count = 0
        edge_count = 0
        id1 = "0"
        for line in lines:
            print("line", line)
            if 'Group User IDs:' in line:
                ids = line.replace("Group User IDs:", "").strip().split(',')
                for groupId1 in ids:
                    for groupId2 in ids:
                        if groupId1 != groupId2:
                            edges_found.append((groupId1.strip(), groupId2.strip()))
            if line[:5] == "User ":
                id1 = line.replace("User ", "").strip()
                user_count = user_count + 1
            if "The user connections are " in line:
                id2s = line.split("The user connections are ")[1].strip().split(',')
                for id2 in id2s[1:]:
                    #print("id1", id1.strip(), "id2", id2.strip())
                    if id1.strip() != id2.strip():
                        edges_found.append((id1.strip(), id2.strip()))
                        edge_count = edge_count + 1
        assert user_count > 50 - 5
        assert len(edges_found) > 50
        assert edge_count > 50
    
    elif method == 'local' or method == 'sequential':
        assert curr_pid is not None, f'{method} method needs curr_pid to parse response'
        new_edges = []
        if include_reason:
            for line in lines:
                pid, reason = line.strip('.').split(',', 1)
                new_edges.append((curr_pid, pid.strip()))
                reasons[pid] = reason.strip()
        else:
            assert len(lines) == 1, f'Response should not be more than one line'
            line = lines[0].replace(',', ' ').replace('.', ' ')
            ids = line.split()
            for pid in ids:
                assert pid.isnumeric(), f'Response should contain ONLY the ID(s)'
                new_edges.append((curr_pid, pid.strip()))
        if num_choices is not None:
            pp = 'people' if num_choices > 1 else 'person'
            assert len(new_edges) == num_choices, f'Choose {num_choices} {pp}'
        edges_found.extend(new_edges)
    
    else:  # iterative-add or iterative-drop
        assert curr_pid is not None, f'{method} method needs curr_pid to parse response'
        if include_reason:
            resp = json.loads(response.strip())
            key = 'new friend' if method == 'iterative-add' else 'dropped friend'
            assert key in resp, f'Missing "{key}" in response'
            pid = str(resp[key])
            action = method.split('-')[1]
            reasons[(pid, action)] = reason
        else:
            assert len(lines) == 1, f'Response should not be more than one line'
            pid = lines[0].strip('.')
            assert len(pid.split()) == 1 and pid.isnumeric(), f'Response should contain only the ID of the person you\'re choosing'
        assert pid.lower() != 'none', 'You must choose one of the IDs in the list'
        edges_found.append((curr_pid, pid))
    
    orig_len = len(edges_found)
    edges_found = set(edges_found)
    if len(edges_found) < orig_len:
        print(f'Warning: {orig_len} edges were returned, {len(edges_found)} are unique')
    
    # check all valid
    valid_nodes = set(G.nodes())
    curr_edges = set(G.edges())
    for id1, id2 in edges_found:
        assert id1 in valid_nodes, f'{id1} is not a node in the network'
        assert id2 in valid_nodes, f'{id2} is not a node in the network'
        if method == 'iterative-drop':
            assert ((id1, id2) in curr_edges) or ((id2, id1) in curr_edges), f'{id2} is not an existing friend'

    # only modify graph at the end
    if method == 'iterative-drop':
        G.remove_edges_from(edges_found)
    else:
        G.add_edges_from(edges_found)
    if include_reason:
        return G, reasons 
    return G
    
    
def generate_network(method, demos_to_include, personas, order, model, mean_choices=None, include_reason=False, 
                     all_demos=False, only_degree=True, num_iter=3, temp=None, verbose=False):
    """
    Generate entire network.
    """
    assert method in {'global', 'global-expressive', 'local', 'sequential', 'iterative'}
    G = nx.Graph()
    G.add_nodes_from(order)
    reasons = {}
    total_num_tries = 0
    total_input_toks = 0
    total_output_toks = 0
    
    if method == 'global':
        system_prompt = get_system_prompt(method, personas, demos_to_include, all_demos=all_demos)
        user_prompt = get_user_prompt(method, personas, order, demos_to_include)
        parse_args = {'method': method, 'G': G}
        G, response, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, update_graph_from_response,
                                                            parse_args, temp=temp, verbose=verbose)
        total_num_tries += num_tries
        total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
        total_output_toks += len(response.split())
    
    elif method == 'global-expressive':
        system_prompt = get_system_prompt(method, personas, demos_to_include, all_demos=all_demos, include_reason=include_reason)
        user_prompt = get_user_prompt(method, personas, order, demos_to_include)
        parse_args = {'method': method, 'G': G}
        #print("system_prompt", system_prompt)
        #print("user_prompt", user_prompt)
        G, response, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, update_graph_from_response,
                                                            parse_args, temp=temp, verbose=verbose, dont_add_errors=True)
        total_num_tries += num_tries
        total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
        total_output_toks += len(response.split())
    
    elif method == 'local' or method == 'sequential':
        order2 = np.random.choice(order, size=len(order), replace=False)  # order of adding nodes
        print('Order of assigning:', order2[:10])
        for node_num, pid in enumerate(order2):
            if mean_choices is None:
                num_choices = None 
            else:
                num_choices = int(min(max(np.random.exponential(mean_choices), 1), 20))
            if node_num < 3:  # for first three nodes, use local
                system_prompt = get_system_prompt('local', personas, demos_to_include, curr_pid=pid,
                                    num_choices=num_choices, include_reason=include_reason, all_demos=all_demos)
                user_prompt = get_user_prompt('local', personas, order, demos_to_include, curr_pid=pid)
            else:  # otherwise, allow local or sequential
                system_prompt = get_system_prompt(method, personas, demos_to_include, curr_pid=pid, 
                    num_choices=num_choices, include_reason=include_reason, all_demos=all_demos, only_degree=only_degree)
                user_prompt = get_user_prompt(method, personas, order, demos_to_include, curr_pid=pid,
                                               G=G, only_degree=only_degree)
            parse_args = {'method': method, 'G': G, 'curr_pid': pid, 'num_choices': num_choices, 'include_reason': include_reason}
            G, response, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, 
                    update_graph_from_response, parse_args, temp=temp, verbose=verbose)
            if include_reason:
                G, pid_reasons = G 
                print(pid, pid_reasons)
                reasons[pid] = pid_reasons
            total_num_tries += num_tries
            total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
            total_output_toks += len(response.split())
            
    else:  # iterative
        # construct local network first 
        order2 = np.random.choice(order, size=len(order), replace=False)  # order of adding nodes
        for pid in order2:
            if mean_choices is None:
                num_choices = None 
            else:
                num_choices = int(max(np.random.exponential(mean_choices), 1))
            system_prompt = get_system_prompt('local', personas, demos_to_include, curr_pid=pid,
                                num_choices=num_choices, include_reason=include_reason, all_demos=all_demos)
            user_prompt = get_user_prompt('local', personas, order, demos_to_include, curr_pid=pid)
            parse_args = {'method': 'local', 'G': G, 'curr_pid': pid, 'num_choices': num_choices, 'include_reason': include_reason}
            G, response, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, 
                    update_graph_from_response, parse_args, temp=temp, verbose=verbose)
            if include_reason:
                G, pid_reasons = G 
                reasons[pid] = pid_reasons
            total_num_tries += num_tries
            total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
            total_output_toks += len(response.split())
        print('Constructed initial network using local method')
        
        for it in range(num_iter):
            print(f'========= ITERATION {it} =========')
            order3 = np.random.choice(order2, size=len(order2), replace=False)  # order of rewiring nodes
            for pid in order3:  # iterate through nodes and rewire
                system_prompt = get_system_prompt('iterative-add', personas, demos_to_include, 
                        curr_pid=pid, G=G, include_reason=include_reason, all_demos=all_demos)
                user_prompt = get_user_prompt('iterative-add', personas, None, demos_to_include, 
                                              curr_pid=pid, G=G)
                parse_args = {'method': 'iterative-add', 'G': G, 'curr_pid': pid, 'include_reason': include_reason}
                G, response_add, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, 
                        update_graph_from_response, parse_args, temp=temp, verbose=verbose)
                if include_reason:
                    G, pid_reasons = G 
                    reasons[pid] = pid_reasons
                total_num_tries += num_tries
                total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
                total_output_toks += len(response_add.split())
                
                friends = list(G.neighbors(pid))
                if len(friends) > 1:
                    system_prompt = get_system_prompt('iterative-drop', personas, demos_to_include, 
                            curr_pid=pid, G=G, include_reason=include_reason, all_demos=all_demos)
                    user_prompt = get_user_prompt('iterative-drop', personas, None, demos_to_include, 
                                                  curr_pid=pid, G=G)
                    parse_args = {'method': 'iterative-drop', 'G': G, 'curr_pid': pid, 'include_reason': include_reason}
                    G, response_drop, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, 
                            update_graph_from_response, parse_args, temp=temp, verbose=verbose)
                    if include_reason:
                        G, pid_reasons = G 
                        reasons[pid] = pid_reasons
                    total_num_tries += num_tries
                    total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
                    total_output_toks += len(response_drop.split())
                else:  
                    assert len(friends) == 1  # must be at least 1 because we just added
                    G.remove_edge(pid, friends[0])
                print(pid, response_add, response_drop)
                
    return G, reasons, total_num_tries, total_input_toks, total_output_toks

def get_save_prefix_and_demos(args):
    """
    Get save prefix and demos to include based on args.
    """
    save_prefix = f'{args.method}_{args.model}'
    demos_to_include = []
    if args.mean_choices != -1:
        assert args.mean_choices > 0
        save_prefix += '_n' + str(args.mean_choices)
    if args.only_interests:
        save_prefix += '_only_interests'
        demos_to_include.append('interests')
    else:
        if args.include_names:
            save_prefix += '_w_names'
            demos_to_include.append('name')        
        demos_to_include.extend(['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation', 'sociability'])
        if args.include_interests:
            save_prefix += '_w_interests'
            demos_to_include.append('interests')
    if args.shuffle_interests:
        assert args.include_interests or args.only_interests
        assert '_INTERESTS_SHUFFLED' in args.persona_fn
        save_prefix += '_INTERESTS_SHUFFLED'
    if args.shuffle_all:
        assert '_ALL_SHUFFLED' in args.persona_fn
        save_prefix += '_ALL_SHUFFLED'
    if args.include_friend_list:
        save_prefix += '_w_list'  # list of friends
    if args.include_reason:
        save_prefix += '_w_reason'
    if args.prompt_all:
        save_prefix += '_prompt_all'
    if args.temp != DEFAULT_TEMPERATURE:
        temp_str = str(args.temp).replace('.', '')
        save_prefix += f'_temp{temp_str}'
    return save_prefix, demos_to_include

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=['global', 'global-expressive', 'local', 'sequential', 'iterative'])
    parser.add_argument('--persona_fn', type=str, default='us_50_w_names_w_interests.json')
    parser.add_argument('--mean_choices', type=int, default=-1)
    parser.add_argument('--include_names', action='store_true')
    parser.add_argument('--include_interests', action='store_true')
    parser.add_argument('--only_interests', action='store_true')
    parser.add_argument('--shuffle_all', action='store_true')
    parser.add_argument('--shuffle_interests', action='store_true')
    parser.add_argument('--include_friend_list', action='store_true')
    parser.add_argument('--include_reason', action='store_true')
    parser.add_argument('--prompt_all', action='store_true')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--num_networks', type=int, default=1)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--temp', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--num_iter', type=int, default=3)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes to use. Defaults to CPU count.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    save_prefix, demos_to_include = get_save_prefix_and_demos(args)
    print('save prefix:', save_prefix)
    
    # Load personas
    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)
    with open(fn) as f:
        personas = json.load(f)
    pids = list(personas.keys())
    print(f'Loaded {len(pids)} personas from {args.persona_fn}')
    
    # Set up parallel processing
    if args.num_processes is None:
        args.num_processes = mp.cpu_count()
    print(f'Using {args.num_processes} processes for parallel network generation')
    
    # Create pool and run parallel processes
    pool = mp.Pool(processes=args.num_processes)
    seeds = range(args.start_seed, args.start_seed + args.num_networks)
    
    # Create partial function with fixed arguments
    generate_network_partial = partial(
        generate_single_network,
        args=args,
        personas=personas,
        pids=pids,
        save_prefix=save_prefix,
        demos_to_include=demos_to_include
    )
    
    # Run processes in parallel and collect results
    stats = pool.map(generate_network_partial, seeds)
    pool.close()
    pool.join()
    
    # Save statistics
    stats_df = pd.DataFrame(stats)
    save_dir = os.path.join(PATH_TO_STATS_FILES, save_prefix)
    if not os.path.exists(save_dir):
        print('Making directory:', save_dir)
        os.makedirs(save_dir)
    
    end_seed = args.start_seed + args.num_networks - 1
    stats_fn = os.path.join(PATH_TO_STATS_FILES, save_prefix, f'cost_stats_s{args.start_seed}-{end_seed}.csv')
    stats_df.to_csv(stats_fn, index=False)