import json
from tqdm import tqdm

def processing(type):
    with open(f'../data/preprocessing/wiki/wiki_{type}_ic.json', 'r') as f:
        data = json.load(f)
    inputs = []
    labels = []
    for ele in tqdm(data):
        inputs.append(ele['inp'])
        labels.append(ele['ref'])
    output = {}
    output['inp'] = inputs
    output['ref'] = labels
    with open(f'../data/preprocessing/wiki/wiki_{type}_sep.json', 'w') as f:
        json.dump(output, f)

def check(type):
    with open(f'../data/preprocessing/wiki/wiki_{type}_ic.json', 'r') as f:
        data = json.load(f)

    print(data['inp'][:5])
    print(len(data['inp']))
    print(data['ref'][:5])
    print(len(data['ref']))

if __name__ == '__main__':
    types = ['train', 'test', 'valid']
    # types = ['valid']
    for type in types:
        # processing(type)
        check(type)