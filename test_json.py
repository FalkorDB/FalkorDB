#! /usr/bin/env python3

import json


with open('cov.json', 'r') as json_file:
    try:
        loaded_json = json.load(json_file)
        print(json.dumps(loaded_json, indent=4))
    except json.JSONDecodeError as e:
        print(f'Error decoding JSON: {e}')
