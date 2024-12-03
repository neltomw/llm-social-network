import os
import argparse
import json
from openai import OpenAI
import re
import pickle
import pandas as pd 
from collections import Counter

from constants_and_utils import *

NAMES_TEMPERATURE = 1.2
PATH_TO_DEMOGRAPHC_DATA = './us_demographics'
RACES = ['White', 'Black', 'American Indian/Alaska Native', 'Asian', 'Native Hawaiian/Pacific Islander', 'Hispanic']
GENDERS = ['Man', 'Woman', 'Nonbinary']
"""
GENERATING PERSONAS PROGRAMMATICALLY
"""
def get_gender_race_age_cdf():
    """
    Get CDF of US distribution, by gender, race, and age.
    From US Census, June 2023
    https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2023/NC-EST2023-ALLDATA.pdf
    https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/national/asrh/ 
    """
    fn = os.path.join(PATH_TO_DEMOGRAPHC_DATA, 'nc-est2023-alldata-r-file07.csv')
    df = pd.read_csv(fn)
    df = df[(df['MONTH'] == 6) & (df['YEAR'] == 2023)]
    assert len(df) == 102

    triplet2ct = {}
    prefixes = ['NHWA', 'NHBA', 'NHIA', 'NHAA', 'NHNA', 'H']
    postfixes = ['MALE', 'FEMALE']
    for _, row in df.iterrows():  # by age
        if row['AGE'] != 999:
            age = row['AGE']
            for pre, race in zip(prefixes, RACES):
                for post, gender in zip(postfixes, GENDERS):
                    triplet2ct[(gender, race, age)] = row[f'{pre}_{post}']
    assert len(triplet2ct) == (101 * len(RACES) * len(GENDERS[:-1])), len(triplet2ct)

    sorted_triplets = sorted(triplet2ct.keys(), key=lambda x: triplet2ct[x], reverse=True)  # sort by largest to smallest triplet
    print(sorted_triplets[:5])
    print(sorted_triplets[-5:])
    counts = [triplet2ct[t] for t in sorted_triplets]
    cdf = np.cumsum(np.array(counts) / np.sum(counts))
    assert np.isclose(cdf[-1], 1.)
    cdf[-1] = 1.
    return sorted_triplets, cdf 


def generate_persona(seed, sorted_triplets, cdf):
    """
    Sample demographics for ONE persona, following joint distributions of US population.
    """
    np.random.seed(seed)
    person = {}
    
    # GENDER, RACE, and AGE - based on US Census
    triplet_rand = np.random.random()
    for triplet, cutoff in zip(sorted_triplets, cdf):
        if triplet_rand <= cutoff:
            gender, race, age = triplet
            person['gender'] = gender 
            person['race/ethnicity'] = race 
            person['age'] = age 
            break
    # add nonbinary - from Pew 
    nonbinary = np.random.random()
    if ((person['age'] < 18) and (nonbinary < 0.03)):
        person['gender'] = 'Nonbinary'
    elif ((person['age'] < 49) and (nonbinary < 0.013)):
        person['gender'] = 'Nonbinary'
    elif nonbinary < 0.001:
        person['gender'] = 'Nonbinary'

    person['sociability'] = np.random.choice(['very introverted','introverted', 'neutral', 'extroverted', 'very extroverted'])
    
    # RELIGION - from Statista
    # https://www.statista.com/statistics/749128/religious-identity-of-adults-in-the-us-by-race-and-ethnicity/
    religion = np.random.random()
    if (person['race/ethnicity'] == 'White'):
        if (religion < 0.49):
            person['religion'] = 'Protestant'
        elif (religion < 0.69):
            person['religion'] = 'Catholic'
        elif (religion < 0.71):
            person['religion'] = 'Jewish'
        elif (religion < 0.72):
            person['religion'] = 'Buddhist'
        else:
            person['religion'] = 'Unreligious'
            
    elif (person['race/ethnicity'] == 'Black'):
        if (religion < 0.68):
            person['religion'] = 'Protestant'
        elif (religion < 0.75):
            person['religion'] = 'Catholic'
        elif (religion < 0.77):
            person['religion'] = 'Muslim'
        else:
            person['religion'] = 'Unreligious'
            
    elif (person['race/ethnicity'] == 'Hispanic'):
        if (religion < 0.26):
            person['religion'] = 'Protestant'
        elif (religion < 0.76):
            person['religion'] = 'Catholic'
        else:
            person['religion'] = 'Unreligious'
    
    elif (person['race/ethnicity'] in ['Asian', 'Native Hawaiian/Pacific Islander']):
        if (religion < 0.16):
            person['religion'] = 'Protestant'
        elif (religion < 0.30):
            person['religion'] = 'Catholic'
        elif (religion < 0.37):
            person['religion'] = 'Muslim'
        elif (religion < 0.44):
            person['religion'] = 'Buddhist'
        elif (religion < 0.59):
            person['religion'] = 'Hindu'
        else:
            person['religion'] = 'Unreligious'
    
    else:
        # from https://www.prri.org/research/2020-census-of-american-religion
        assert person['race/ethnicity'] == 'American Indian/Alaska Native'
        if (religion < 0.47):
            person['religion'] = 'Protestant'
        elif (religion < 0.58):
            person['religion'] = 'Catholic'
        elif (religion < 0.60):
            person['religion'] = 'Christian'
        else:
            person['religion'] = 'Unreligious'

    # POLITICAL AFFILIATION - from Pew
    # https://www.pewresearch.org/politics/2024/04/09/partisanship-by-race-ethnicity-and-education/#partisanship-by-race-and-gender 
    politics = np.random.random()
    person['political affiliation'] = 'Independent'
    if person['race/ethnicity'] == 'White':
        if person['gender'] == 'Man':
            if politics < 0.6:
                person['political affiliation'] = 'Republican'
            elif politics < 0.99:
                person['political affiliation'] = 'Democrat'
        else:
            if politics < 0.53:
                person['political affiliation'] = 'Republican'
            elif politics < 0.96:
                person['political affiliation'] = 'Democrat'
    elif person['race/ethnicity'] == 'Black':
        if person['gender'] == 'Man':
            if politics < 0.15:
                person['political affiliation'] = 'Republican'
            elif politics < 0.96:
                person['political affiliation'] = 'Democrat'
        else:
            if politics < 0.10:
                person['political affiliation'] = 'Republican'
            elif politics < 0.94:
                person['political affiliation'] = 'Democrat'
    elif person['race/ethnicity'] == 'Hispanic':
        if person['gender'] == 'Man':
            if politics < 0.39:
                person['political affiliation'] = 'Republican'
            elif politics < 1:
                person['political affiliation'] = 'Democrat'
        else:
            if politics < 0.32:
                person['political affiliation'] = 'Republican'
            elif politics < 0.92:
                person['political affiliation'] = 'Democrat'
    elif person['race/ethnicity'] in ['Asian', 'Native Hawaiian/Pacific Islander']:
        if person['gender'] == 'Man':
            if politics < 0.39:
                person['political affiliation'] = 'Republican'
            elif politics < 1:
                person['political affiliation'] = 'Democrat'
        else:
            if politics < 0.36:
                person['political affiliation'] = 'Republican'
            elif politics < 1:
                person['political affiliation'] = 'Democrat'
    else:
        # https://www.brookings.edu/articles/native-americans-support-democrats-over-republicans-across-house-and-senate-races/
        assert person['race/ethnicity'] == 'American Indian/Alaska Native'
        if politics < 0.4:
            person['political affiliation'] = 'Republican'
        elif politics < 0.96:
            person['political affiliation'] = 'Democrat'
    return person


def convert_persona_to_string(persona, demos_to_include, pid=None):
    """
    Convert pid (an int) and persona (a dictionary) into a string.
    """
    if pid is None:
        s = ''
    else:
        s = f'{pid}. '
    if 'name' in demos_to_include:
        name = ' '.join(persona['name'])
        s += f'{name} - '
    for pos, demo in enumerate(demos_to_include):
        if demo != 'name':
            if demo == 'age':
                s += f'age {persona[demo]}, '  # specify age so GPT doesn't get number confused with ID
            elif demo == 'interests' and pos > 0:  # not first demo
                s += f'interests include: {persona[demo]}, '
            else:
                s += f'{persona[demo]}, '
    s = s[:-2]  # remove trailing ', '
    return s  


def assign_persona_to_model(persona, demos_to_include):
    """
    Describe persona in second person: "You are..."
    """
    s = 'You are '
    persona_str = convert_persona_to_string(persona, demos_to_include)
    if 'name' in demos_to_include:
        s += persona_str
    else:
        first_demo = demos_to_include[0]
        if first_demo in ['gender', 'political affiliation']:  # noun
            article = 'an' if persona[first_demo].lower()[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
            s += article + ' ' + persona_str
        elif first_demo in ['race/ethnicity', 'age', 'religion','sociability']:  # adjective
            s += persona_str 
        else:
            assert first_demo == 'interests'
            s += 'interested in ' + persona_str
    return s


def generate_names(personas, demos, model, verbose=False):
    """
    Generate names, using GPT, for a list of personas.
    """
    for nr in personas:
        prompt = 'Generate a name for someone with the following demographics:\n'
        for demo in demos:
            prompt += f'{demo}: {personas[nr][demo]}\n'
        prompt += 'Answer by providing ONLY their first and last name, in the format "FIRSTNAME LASTNAME".'
        name, _, _ = repeat_prompt_until_parsed(model, None, prompt, parse_name_response, {}, max_tries=3,
                                                temp=NAMES_TEMPERATURE,  verbose=verbose)
        personas[nr]['name'] = name
        print(convert_persona_to_string(personas[nr], demos, pid=nr), personas[nr]['name'])
    return personas

def parse_name_response(response):
    words = re.findall('[a-zA-Z]+', response)
    if len(words) == 2:
        return words[0].capitalize(), words[1].capitalize()
    else:
        raise Exception('Response contains more than two words')


def generate_interests(personas, demos, model, verbose=False):
    """
    Generate interests, using GPT, for a list of personas.
    """
    for nr in personas:
        prompt = f'In 10-25 words, describe the varied interests that the following person might have:\n'
        rand_order = np.random.choice(len(demos), replace=False, size=len(demos))  # shuffle order of demographics
        for idx in rand_order:
            demo = demos[idx]
            prompt += f'{demo}: {personas[nr][demo]}\n'
        prompt += 'Answer by providing ONLY their interests. Do not include filler like "She enjoys" or "He has a keen interest in".'

        prompt += """
Here are some examples of possible interests:
Reading, Writing, Drawing, Painting, Photography, Gardening, Cooking, Baking, Traveling, Hiking, Running, Cycling, Swimming, Camping, Fishing, Hunting, Rock climbing, Surfing, Skiing, Snowboarding, Skateboarding, Yoga, Meditation, Weightlifting, Pilates, Dancing, Singing, Playing guitar, Playing piano, Playing violin, Playing drums, Playing flute, Playing ukulele, Playing chess, Solving puzzles, Playing video games, Board games, Card games, Collecting stamps, Collecting coins, Collecting antiques, Collecting comic books, Collecting action figures, Collecting postcards, Collecting toys, Knitting, Crocheting, Sewing, Quilting, Embroidery, Origami, Calligraphy, Scrapbooking, Woodworking, Metalworking, Pottery, Sculpting, Model building, Home brewing, Wine tasting, Beer tasting, Coffee brewing, Tea appreciation, Cooking international cuisines, Gardening succulents, Interior decorating, DIY home improvement, Antique restoration, Restoring furniture, Making jewelry, Making candles, Making soap, Watching movies, Watching TV series, Watching documentaries, Watching anime, Writing poetry, Writing novels, Writing scripts, Blogging, Vlogging, Podcasting, Public speaking, Stand-up comedy, Acting, Directing, Producing films, Photography editing, Graphic design, Digital art, Animation, 3D modeling, Web design, Programming, Learning new languages, Studying history, Studying science, Astronomy, Astrology, Biology, Physics, Chemistry, Geology, Meteorology, Archaeology, Anthropology, Psychology, Sociology, Political science, Economics, Business management, Entrepreneurship, Marketing, Sales strategy, Investing, Trading stocks, Cryptocurrency, Real estate, Personal finance, Budgeting, Minimalism, Sustainability, Environmental activism, Animal rights activism, Human rights activism, Volunteering, Charity work, Fundraising, Mentoring, Tutoring, Teaching, Parenting, Pet care, Dog training, Cat care, Horseback riding, Birdwatching, Wildlife photography, Wildlife conservation, Marine biology, Scuba diving, Snorkeling, Free diving, Kayaking, Canoeing, Sailing, Rowing, Powerboating, Motorcycling, Driving, Car racing, Bicycle touring, Backpacking, Trail running, Ultra running, Triathlon, Marathon running, Powerlifting, CrossFit, Boxing, Martial arts, Judo, Karate, Taekwondo, Brazilian Jiu-Jitsu, Krav Maga, Fencing, Archery, Shooting sports, Golf, Tennis, Table tennis, Badminton, Squash, Racquetball, Cricket, Baseball, Softball, Basketball, Football, Soccer, Rugby, Hockey, Field hockey, Ice hockey, Lacrosse, Volleyball, Beach volleyball, Bowling, Darts, Billiards, Kite flying, Kiteboarding, Paragliding, Hang gliding, Skydiving, Bungee jumping, Parkour, Urban exploration, Travel photography, Historical site visits, Museum hopping, Theme park visits, Escape rooms, Magic tricks, Illusion shows, Playing with drones, Drone photography, RC cars, RC planes, RC boats, Geocaching, Treasure hunting, Metal detecting, Prospecting, Stargazing, Astrophotography, Meteor shower watching, Northern lights watching, Camping under the stars, Storytelling, Improvisation, Debating, Philosophy, Ethics, Religion, Spirituality, Tarot reading, Crystal healing, Reiki, Acupuncture, Herbal medicine, Holistic health, Nutrition, Dieting, Weight management, Fitness tracking, Personal training, Sports coaching, Event planning, Hosting parties, Social media management, Online community building, E-sports, Speedrunning, Gaming live streams, Tech reviews, Unboxing videos, ASMR, Sound design, Music production, Beatboxing, DJing, Singing in a choir, Musical theater, Opera, Ballet, Classical music, Jazz music, Blues music, Country music, Hip-hop, Rap, Pop music, Rock music, Heavy metal, Punk rock, Folk music, World music, Reggae, EDM, Trance music, Ambient music, Lo-fi beats, Collecting vinyl records, Watching live concerts, Attending music festivals, Event photography, Fashion photography, Street photography, Nature photography, Macro photography, Film photography, Darkroom development, Digital photo editing, Restoring old photos, Wedding photography, Pet photography, Food photography, Interior photography, Landscape photography, Cityscape photography, Architectural photography, Abstract photography, Panoramic photography, Night photography, Aerial photography, Underwater photography, Forensic photography, Documentary photography, Journalism, Investigative reporting, Editing, Proofreading, Translation, Transcription, Copywriting, Technical writing, Grant writing, Science writing, Medical writing, Business writing, Legal writing, Screenwriting, Playwriting, Ghostwriting, Creative writing, Writing children's books, Writing fantasy novels, Writing science fiction, Writing historical fiction, Writing thrillers, Writing romance novels, Writing self-help books, Writing memoirs, Writing essays, Writing blogs, Creating memes, Posting on social media, Influencing, Public relations, Marketing campaigns, Advertising, SEO optimization, Analytics, Data science, Machine learning, AI development, Natural language processing, Big data, Cloud computing, Cybersecurity, Game development, App development, Web development, Blockchain, Augmented reality, Virtual reality, IoT, Quantum computing, Space exploration, Rocket science, Satellite technology, Nanotechnology, Bioinformatics, Genetics, Genomics, Proteomics, Epigenetics, Synthetic biology, Cognitive science, Neuroscience, Behavioral science, Social psychology, Organizational behavior, Learning theories, Educational psychology, Pedagogy, Curriculum design, Ed-tech, Digital literacy, Open-source projects, Cryptography, Algorithm design, Simulation, 3D printing, Materials science, Renewable energy, Sustainable agriculture, Urban planning, Transport design, Climate activism, Carbon offsetting, Green tech, Energy storage, Battery technology, Electric vehicles, Autonomous vehicles, Flying cars, Deep-sea exploration, Oceanography, Coral reef studies, Aquaculture, Desalination, Water purification, Agronomy, Agroforestry, Hydroponics, Vertical farming, Perennial crops, Soil science, Permaculture, Agroecology, Ecosystem restoration, Rewilding, Urban rewilding, Microbiome research, Bacteriology, Virology, Immunology, Pathology, Pharmacology, Toxicology, Epidemiology, Biostatistics, Health informatics, Telemedicine, Healthcare policy, Health economics, Public health, Medical anthropology, Forensic science, Criminology, Law enforcement, Emergency response, Firefighting, Search and rescue, Military strategy, Defense technology, Peacekeeping, Humanitarian aid, Crisis management, Disaster recovery, Refugee support, Global development, Microfinance, Social entrepreneurship, Community building, Social capital, Civic engagement, Participatory design, Citizen science, Crowdsourcing, Open innovation, Human-centered design, UX/UI design, Game theory, Behavioral economics, Financial modeling, Risk management, Actuarial science, Accounting, Auditing, Tax planning, Business strategy, Operations management, Supply chain logistics, Lean manufacturing, Process improvement, Industrial engineering, Engineering management, Product design, Market research, Consumer behavior, Retail trends, E-commerce, International trade, Economic policy, Development economics, Public administration, Urban governance, Regional planning, Housing policy, Transportation policy, Energy policy, Food policy, Education policy, Social policy, Environmental policy, Labor policy, Tax policy, Trade policy, Technology policy, Media policy, Cultural policy, Security policy, Peace studies, Conflict resolution, Negotiation, Diplomacy, International relations, Geopolitics, Global health, Global education, Global finance, Global justice, Ethnography, Qualitative research, Quantitative research, Mixed-methods research, Data visualization, Information design, Knowledge management, Organizational learning, Change management, Leadership development, Team building, Employee engagement, Cultural competence, Diversity and inclusion, Work-life balance, Remote work, Digital transformation, Innovation ecosystems, Disruptive technology, Future studies, Scenario planning, Strategic foresight, Resilience planning, Risk assessment, Crisis communication, Brand management, Corporate social responsibility, Sustainability reporting, Ethical leadership, Servant leadership, Transformational leadership, Authentic leadership, Adaptive leadership, Emotional intelligence, Social intelligence, Cognitive intelligence, Artificial intelligence, Human-machine interaction, Cyber-physical systems, Digital twin technology, Edge computing, Fog computing, Serverless computing, DevOps, Continuous integration, Software testing, Quality assurance, Reliability engineering, Performance engineering, Systems architecture, Enterprise architecture, Solution architecture, Cloud architecture, Distributed systems, Concurrency, Parallel computing, Grid computing, Cluster computing, High-performance computing, Numerical simulation, Computational biology, Computational chemistry, Computational physics, Computational linguistics, Computational neuroscience, Neural networks, Deep learning, Reinforcement learning, Transfer learning, Few-shot learning, Self-supervised learning, Semi-supervised learning, Unsupervised learning, Graph learning, Time-series analysis, Speech recognition, Natural language generation, Image recognition, Video recognition, Emotion recognition, Pattern recognition, Facial recognition, Object detection, Pose estimation, Motion capture, Scene understanding, Semantic segmentation, Instance segmentation, Keypoint detection, Optical character recognition, Machine translation, Text summarization, Text-to-speech, Speech-to-text, Text mining, Opinion mining, Sentiment analysis, Topic modeling, Named entity recognition, Knowledge graph construction, Ontology engineering, Semantic web, Linked data, Open data, Crowdsourced data, Big data analytics, Data cleaning, Data wrangling, Feature engineering, Model selection, Hyperparameter tuning, Ensemble methods, Boosting algorithms, Bayesian inference, Probabilistic programming, Monte Carlo methods, Markov models, Game playing AI, Robotics, Autonomous systems, Self-driving cars, Space robotics, Medical robotics, Agricultural robotics, Underwater robotics, Drone technology, Exoskeletons, Bionics, Prosthetics, Assistive technology, Wearable devices, Augmented reality glasses, Virtual reality headsets, Haptics, Brain-computer interfaces, Neural prosthetics, Neurotechnology, Digital therapeutics, Telehealth, Mobile health, Precision medicine, Personalized medicine, Pharmacogenomics, Therapeutic innovation, Medical imaging, Diagnostic imaging, 3D printing in healthcare, Tissue engineering, Regenerative medicine, Stem cell research, Organoid research, Bioengineering, Synthetic organs, Artificial intelligence in healthcare, Medical decision support systems, Clinical trials, Translational research, Health outcomes research, Comparative effectiveness research, Implementation science, Global health equity, Healthcare accessibility, Health systems strengthening, Health policy advocacy, Patient advocacy, Digital health startups, Health tech entrepreneurship, Social determinants of health, Health equity, Global health diplomacy, Healthcare innovation
"""
        interests, _, _ = repeat_prompt_until_parsed(model, None, prompt, parse_interest_response, {}, max_tries=3,
                                                     temp=NAMES_TEMPERATURE, verbose=verbose)
        personas[nr]['interests'] = interests
        print(convert_persona_to_string(personas[nr], demos + ['interests'], pid=nr))
    return personas
    
def parse_interest_response(response):
    response = response.strip().strip('.')
    toks = response.split()
    if toks[0].lower() in ['he', 'she', 'they']:
        raise Exception('Do not include filler. Provide ONLY their interest as one phrase.')
    if len(toks) > 100:
        raise Exception('Interests are too long')
    return response

def get_interest_embeddings(persona_fn, model='text-embedding-3-small'):
    """
    Get text embeddings for each generated interest.
    """
    fn = os.path.join(PATH_TO_TEXT_FILES, persona_fn)
    with open(fn) as f:
        personas = json.load(f)
    save_name = os.path.join(PATH_TO_TEXT_FILES, f'{persona_fn[:-5]}_{model}.pkl')
    print('Will save embeddings in ', save_name)
    
    embs = {}
    for key in personas:
        text = personas[key]['interests']
        emb = CLIENT.embeddings.create(input = [text], model=model).data[0].embedding
        embs[key] = np.array(emb)
        print(key)
    with open(save_name, 'wb') as f:
        pickle.dump(embs, f)
    return embs


def get_interest_similarities(demo, personas, embs, min_sims=30):
    """
    Compute cosine similarity between interests for pairs from same group
    vs. different group.
    demo: demographic variable, eg, 'gender', 'race/ethnicity'

    Cosine similarity is recommended by OpenAI for measuring distance:
    # We recommend cosine similarity. The choice of distance function typically doesn’t matter much.
    # OpenAI embeddings are normalized to length 1, which means that:
    # Cosine similarity can be computed slightly faster using just a dot product
    """
    assert set(personas.keys()) == set(embs.keys())
    # n = len(personas.keys())
    # all_embs = np.concatenate([embs[k].reshape(1, -1) for k in embs], axis=0)
    # print(all_embs.shape)
    # assert len(all_embs) == n
    # all_sims = all_embs @ all_embs.T 
    # all_sims = np.triu(all_sims, 1)  # zero out diagonal and bottom triangle
    # all_sims = all_sims.flatten()
    # all_sims = all_sims[~np.isclose(all_sims, 0)]  # remove 0 entries 
    # assert len(all_sims) == (n*(n-1))/2
    # avg_sim = np.mean(all_sims)
    # print('Avg similarity:', avg_sim)

    vals = [personas[k][demo] for k in personas]
    val_counts = Counter(vals)
    print(val_counts)
    unique_vals = [v for (v, _) in val_counts.most_common()]  # in order from most to least common
    group2embs = {v:[] for v in unique_vals}  # map group (e.g., 'woman') to interest embedding 
    for key in personas:
        v = personas[key][demo]
        group2embs[v].append(embs[key])
    
    same_group = []
    diff_group = []
    pair_to_sims = {}
    for id, v1 in enumerate(unique_vals):
        embs1 = np.array(group2embs[v1])
        n1 = len(embs1)
        # compute similarity within group
        sims = embs1 @ embs1.T 
        sims = np.triu(sims, 1)  # zero out diagonal and bottom triangle
        sims = sims.flatten() #  / avg_sim
        sims = sims[~np.isclose(sims, 0)]  # remove 0 entries 
        assert len(sims) == (n1*(n1-1))/2
        same_group.append(sims)
        if len(sims) >= min_sims:
            pair_to_sims[(v1, v1)] = sims
        else:
            print(f'Not saving {v1}, {v1}, only {len(sims)} pairs')

        # compute similarity with other groups
        if id < len(unique_vals)-1:
            for v2 in unique_vals[id+1:]:
                embs1 = np.array(group2embs[v1])
                embs2 = np.array(group2embs[v2])
                sims = (embs1 @ embs2.T).flatten() #  / avg_sim
                diff_group.append(sims)
                if len(sims) >= min_sims:
                    pair_to_sims[(v1, v2)] = sims
                else:
                    print(f'Not saving {v1}, {v2}, only {len(sims)} pairs')
    
    same_group = np.concatenate(same_group)
    diff_group = np.concatenate(diff_group)
    return same_group, diff_group, pair_to_sims

def make_demographic_scatter_plot(demo, personas, x, y, save_plot=True, interests_args='', group2color=None, cutoff=1):
    assert len(x) == len(y)
    assert len(personas) == len(x)
    if demo == 'age':
        plt.figure(figsize=(4.5,4))
        c = [personas[k]['age'] for k in personas]
        plt.scatter(x, y, c=c)
        plt.colorbar()
    else:
        plt.figure(figsize=(4,4))
        group2idx = {}
        for nr in personas:
            v = personas[nr][demo]
            group2idx[v] = group2idx.get(v, []) + [int(nr)]
        
        group_order = sorted(group2idx.keys(), key=lambda x: len(group2idx[x]), reverse=True)
        for group in group_order:
            idx = group2idx[group]
            if len(idx) >= cutoff:
                x_gr = np.array(x)[idx]
                y_gr = np.array(y)[idx]
                if group2color is not None:
                    plt.scatter(x_gr, y_gr, label=group, color=group2color[group])
                else:
                    plt.scatter(x_gr, y_gr, label=group)
            else:
                print('Dropping', group)
        plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(alpha=0.2)
    plt.title(demo.capitalize(), fontsize=16)
    if save_plot:
        fn = f'plots/interests{interests_args}-viz-{demo[:4]}.pdf'
        print(fn)
        plt.savefig(fn, bbox_inches='tight')
    else:
        plt.show()

def parse_reason(model, reason, demos_to_include, verbose=False):
    """
    Classify free-text reason into list of demographic variables.
    """
    def parse_classification(response, demos_to_include):
        if response.startswith('Answer:'):
            response = response[len('Answer:'):]
        groups = response.split(',')
        kept_groups = []
        for g in groups:
            g = g.lower().strip()
            assert g in demos_to_include
            kept_groups.append(g)
        return kept_groups

    system = 'You will be given a reason why someone is friends with someone else. Select which demographic variables, out of {'
    system += ', '.join(demos_to_include)
    system += '}, are provided as the reason for friendship. You must select at least one and can select multiple. Format your answer as a comma-separated list.'
    system += '\n\nExample: "I appreciate the diversity in age and race but also share the same political affiliation as a Democrat"'
    system += '\nAnswer: political affiliation'
    system += '\n\nExample: "As a fellow unreligious individual and Democrat, I feel a connection with this young man"'
    system += '\nAnswer: religion, political affiliation, age, gender'
    if verbose:
        print(system)
    try:
        parse_out, _, _ = repeat_prompt_until_parsed(model, system, reason, parse_classification,
                                        {'demos_to_include': demos_to_include}, max_tries=3, temp=DEFAULT_TEMPERATURE, verbose=False)
        return parse_out
    except:
        print('Could not classify:', reason)
        return None 

def parse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    # Add arguments
    parser.add_argument('number_of_people', type=int, help='How many people would you like to generate?')
    parser.add_argument('save_name', type=str, help='What is the name of the file where you would like to save the personas?')
    parser.add_argument('--include_names',  action='store_true', help='Would you like to add names to the personas?')
    parser.add_argument('--include_interests',  action='store_true', help='Would you like to add interests to the personas?')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Which model would you like to use for generating names/interests?')

    args = parser.parse_args()    
    return args


if __name__ == '__main__':
    args = parse()
    # generate personas with GPT
    n = args.number_of_people
    save_name = args.save_name
    demos_to_include = ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation', 'sociability']
    # demos_to_include = ['gender', 'race/ethnicity']  # TEMPORARY

    # get distributions from US Census data
    sorted_triplets, cdf = get_gender_race_age_cdf()

    # generate persona demographics
    personas = {}
    for i in range(n):
        personas[i] = generate_persona(i, sorted_triplets, cdf)
    
    # generate names
    if args.include_names:
        save_name += '_w_names'
        personas = generate_names(personas, demos_to_include, args.model)

    # generate interests
    if args.include_interests:
        save_name += '_w_interests'
        personas = generate_interests(personas, demos_to_include, args.model)

    # save json
    fn = os.path.join(PATH_TO_TEXT_FILES, save_name + '.json')
    with open(fn, 'w') as f:
        json.dump(personas, f)

    # if args.include_names:
    #     fn = fn[:-5] + "_with_names.json"

    #     personas = generate_names(personas, demos_to_include, args.model)

    #     # count all unique last names in personas[person]['name']
    #     counts = {}
    #     personas_for_saving = {}
    #     for person in personas:
    #         last_name = personas[person]['name'].split(' ')[1]
    #         if last_name in counts:
    #                 counts[last_name] += 1
    #         else:
    #                 counts[last_name] = 1
    #         personas_for_saving[f'{personas[person]["name"].replace(" ", "-")}'] = personas[person]
    #         del personas[person]['name']

    #     # save to json
    #     with open(fn, 'w') as f:
    #         json.dump(personas_for_saving, f)

    #     personas = personas_for_saving

    #     # print counts in sorted order
    #     print(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    # if args.include_interests:
    #     fn = fn[:-5] + "_with_interests.json"

    #     # save json file
    #     personas = generate_interests(personas, demos_to_include, args.model)
    #     with open(fn, 'w') as f:
    #         json.dump(personas, f)

    # pass arguments: # of people, save path
#
#    fn = os.path.join(PATH_TO_TEXT_FILES, 'programmatic_personas.txt')
#    personas, demo_keys = load_personas_as_dict(fn) -- assert lines[0].startswith('Name - ')
#    personas = generate_names(personas)
#    print(personas)
    
    
