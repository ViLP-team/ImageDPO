import os
import base64
import requests
from IPython import embed
import glob
import json
from PIL import Image
import io
import pandas as pd
import argparse
import numpy as np
import spacy

nlp = spacy.load('en_core_web_md')  # Load the medium or larger model for better vector similarity

# def are_synonyms_spacy(word1, word2, similarity_threshold=0.95):
#     """Check if two words are synonyms using spaCy's similarity function."""
#     doc1, doc2 = nlp(word1), nlp(word2)
#     if doc1.similarity(doc2) > similarity_threshold:
#         print(f"Similarity between '{word1}' and '{word2}': {doc1.similarity(doc2)}")
#     return doc1.similarity(doc2) > similarity_threshold

def normalize_output(output):
    """
    Normalize the output to a standard format.
    Includes number_mapping, synonym mapping and plural-singular conversions.
    """
    
    # Convert word numbers to digits
    number_mapping = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    }
    
    # Synonym mapping
    synonym_mapping = {
        'refrigerator': 'fridge',
        'stove': 'oven',
        'alligator': 'crocodile',
        'porpoise': 'dolphin',
        'automobile': 'car',
        'nyc': 'new york city',
        'la': 'los angeles',
        'usa': 'united states',
        'co2': 'carbon dioxide',
        'o2': 'oxygen',
        'n2': 'nitrogen',
        'h2o': 'water',
        'tortoise': 'turtle',
        'motorbike': 'motorcycle',
        'cellphone': 'phone',
        'telephone': 'phone',
        'pc': 'computer',
        'tv': 'television',
        'tap': 'faucet',
        'aeroplane': 'airplane',
        'cubic': 'cube',
        'cubical': 'cube',
        'cubes': 'cube',
        'cuboids': 'cube',
        'cuboid': 'cube',
        'square': 'cube',
        'squares': 'cube',
        'striped': 'stripes',
        'checkered': 'checkerboard',
        'polka-dots': 'spots',
        'dalmatian': 'dog',
        'triangular': 'triangle',
        'circular': 'round',
        'circle': 'round',
        'circles': 'round',
        'spherical': 'round',
        'spheres': 'round',
        'triangles': 'triangle',
        'logs': 'wood',
        'zigzag': 'curved',
        'hexagonal': 'hexagon',
        'bud': 'flower',

        'hippopotamus': 'hippo',
        'rhinoceros': 'rhino',
        'bike': 'bicycle',
        'stove': 'oven',
        'schoolbus': 'bus',
        'boat': 'ship',
        'sailboat': 'ship',
        'airship': 'ship',

        'donut': 'torus',
        'donuts': 'torus',
        'wallaby': 'kangaroo',
        'teacup': 'cup',
        'teapot': 'kettle',
        'rooster': 'chicken',
        'roosters': 'chicken',
        'raven': 'crow',
        'vineyard': 'vine',
        'bushe': 'bush',
        'crystal': 'glass',
        'hay': 'straw',
        'fireplace': 'oven',
        'co₂': 'carbondioxide',
        'aircondition': 'AC',
        'airconditioner': 'AC',
        'air-conditioner': 'AC',
        't-rex': 'dinosaur',
        'trex': 'dinosaur',
        'man': 'person',
        'woman': 'person',
        'people': 'person',
        'men': 'person',
        'women': 'person',
        'clocktower': 'bigben',
        'multicolored': 'rainbow',
        'thatch': 'straw',
        'underwater': 'water',
        'plane': 'airplane',
        'goggles': 'glasses',
        'night-vision': 'glasses',
        'blossoms': 'flower',
        'brush': 'eraser',
        'serpent': 'snake',
        'dots': 'spots',
        'binoculars': 'glasses',
        'slippers': 'shoe',
        'slipper': 'shoe',
        'pillow': 'cushion',
        'hexagonal': 'hexagon',
        'hexagons': 'hexagon',
        'ukulele': 'guitar',
        'cello': 'violin',
        'America': 'USA',
        'steel': 'metal',
        'cucumber': 'pickle',
        'galaxy': 'space',
        'cup': 'teacup',
        'underwater': 'sea',
        'ocean': 'sea',
        'faceted': 'diamond',
        'jewelry': 'diamond',
        'jewelries': 'diamond',
        'backpack': 'bag',
        'squid': 'octopus',
        'kitten': 'cat',
        'octagonal': 'octagon',
        'candy': 'lolipop',
        'pipeline': 'pipe',
        'dragonfruit': 'pitaya',
    }
    
    # Plural-singular mapping
    plural_singular_mapping = {
        'butterflies': 'butterfly',
        'bees': 'bee',
        'ants': 'ant',
        'wasps': 'wasp',
        'kangaroos': 'kangaroo',
        'koalas': 'koala',
        'wombats': 'wombat',
        'trees': 'tree',
        'books': 'book',
        'goats': 'goat',
        'squirrels': 'squirrel',
        'rabbits': 'rabbit',
        'pandas': 'panda',
        'giraffes': 'giraffe',
        'lions': 'lion',
        'tigers': 'tiger',
        'cows': 'cow',
        'horses': 'horse',
        'cats': 'cat',
        'dogs': 'dog',
        'whales': 'whale',
        'sharks': 'shark',
        'dolphins': 'dolphin',
        'flowers': 'flower',
        'leaves': 'leaf',
        'knives': 'knife',
        'wolves': 'wolf',
        'mice': 'mouse',
        'geese': 'goose',
        'children': 'child',
        'teeth': 'tooth',
        'feet': 'foot',
        'fungi': 'fungus',
        'stimuli': 'stimulus',
        'media': 'medium',
        'octopi': 'octopus',
        'cacti': 'cactus',
        'diamonds': 'diamond',
        'bricks': 'brick',  
        'flame': 'fire',
        'winds': 'wind',
        'wheels': 'wheel',
        'chickens': 'chicken',
        'fireflies': 'firefly',
        'beaks': 'beak',
        'needles': 'needle',
        'spinners': 'spinner',
        'clouds': 'cloud',
        'earthquakes': 'earthquake',
        'seals': 'seal',
        'pandas': 'panda',
        'pencils': 'pencil',
        'petals': 'petal',
        'forks': 'fork',
        'petals': 'petal',
        'seahorses': 'seahorse',
        'keys': 'key',
        'carrots': 'carrot',
        'crayons': 'crayon',
        'skyscrapers': 'skyscraper',
        'birds': 'bird',
        'bicycles': 'bicycle',
        'watches': 'watch',
        'lemons': 'lemon',
        'spinners': 'spinner',
        'pipes': 'pipe',
        'spinnerets': 'spinneret',
        'bubbles': 'bubble',
        'camels': 'camel',
        'stripes': 'stripe',
        'pandas': 'panda',
        'mice': 'mouse',
        'lungs': 'lung',
        'gills': 'gill',
        'diamonds': 'diamond',
        'feathers': 'feather',
        'scales': 'scale',
        'lollipops': 'lolipop',
        'lollipop': 'lolipop',
        'lolipops': 'lolipop',
        'drums': 'drum',
        'ropes': 'rope',
        'shoes': 'shoe',
    }

    output = str(output).lower().strip()
    if len(output) != 0:
        if output[-1] == '.':
            output = output[:-1]
    
    output = number_mapping.get(output, output)
    output = synonym_mapping.get(output, output)
    output = plural_singular_mapping.get(output, output)

    return output
    
def compare_single_output(our, gt):
    output = normalize_output(our)
    gt = normalize_output(gt)
    if output == gt:
        return True

    if output == 'none':
        return False
    
    # return are_synonyms_spacy(output, gt)
    return False

def compare_outputs(our_results, gt_results):
    if len(our_results) != len(gt_results):
        print("Warning: Lists have different lengths.")
        return 0, len(our_results), len(gt_results)
    
    matches = 0
    matches_details = []
    total = len(our_results)
    
    for our, gt in zip(our_results, gt_results):
        if compare_single_output(our, gt):
        # if our == gt:
            matches += 1
            matches_details.append(True)
        else:
            matches_details.append(False)
    
    return matches, total, matches_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str, default="answer.jsonl")
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    with open(args.result_file, 'r') as f:
        results = [json.loads(line) for line in f]
    our_results = []
    gt_results = []

    for result in results:
        if "gen_answers" in result and "gt_answers" in result:
            our_results.extend(result["gen_answers"])
            gt_results.extend(result["gt_answers"])

    matches, total_num, matches_details = compare_outputs(our_results, gt_results)
    output_info = (
        f"Matching outputs: {matches}/{total_num}\n"
        f"Percentage match: {matches/total_num*100:.2f}%\n"
        f"each question answer accuracy: {np.mean(np.array(matches_details).reshape([-1,3]),0)}\n"
        f"each question answer correct counting [0, 1, 2, 3]: "
        f"{np.sum(np.sum(np.array(matches_details).reshape([-1,3]),1)==0)}, "
        f"{np.sum(np.sum(np.array(matches_details).reshape([-1,3]),1)==1)}, "
        f"{np.sum(np.sum(np.array(matches_details).reshape([-1,3]),1)==2)}, "
        f"{np.sum(np.sum(np.array(matches_details).reshape([-1,3]),1)==3)} \n"
        f'final score: {np.mean(np.mean(np.array(matches_details).reshape([-1,3]), 0)[1:]) * 100:.2f}'
    )
    print(output_info)
    if args.output_file is not None:
        with open(args.output_file, 'a') as f:
            f.write(args.result_file + "\n")
            f.write(output_info + "\n")
