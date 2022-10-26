'''
Tihm and Minder recipe.
'''

import pandas as pd

import dcarte
from dcarte.local import LocalDataset
from dcarte.utils import process_transition







########## processing functions



def process_add_tihm_motion(self):

    tihm = self.datasets['legacy'][['patient_id', 'start_date', 'location_name']]
    minder = self.datasets['base']

    concat_df = pd.concat([tihm, minder], ignore_index=True).drop_duplicates()

    return concat_df





def process_add_tihm_transitions(self):
    motion = self.datasets['motion']
    motion = process_transition(motion, ['patient_id'], 'start_date', 'location_name')
    return motion.reset_index()






########## creating datasets

def create_tihm_and_minder_datasets():
    domain = 'tihm_and_minder'
    module = 'tihm_and_minder'

    parent_datasets = {
        'motion': [['motion', 'legacy'], ['motion', 'base']],
        'transitions': [['motion', 'tihm_and_minder']],
        

    }
    
    module_path = __file__
    print('adding Tihm to motion data')
    LocalDataset(
        dataset_name='motion',
        datasets={d[1]: dcarte.load(*d) for d in parent_datasets['motion']},
        pipeline=['process_add_tihm_motion'], 
        domain=domain,
        module=module,
        module_path=module_path,
        reload=True,
        dependencies=parent_datasets['motion'])

    print('adding Tihm to transitions data')
    LocalDataset(
        dataset_name='transitions',
        datasets={d[0]: dcarte.load(*d) for d in parent_datasets['transitions']},
        pipeline=['process_add_tihm_transitions'], 
        domain=domain,
        module=module,
        module_path=module_path,
        reload=True,
        dependencies=parent_datasets['transitions'])



if __name__ == '__main__':
    create_tihm_and_minder_datasets()