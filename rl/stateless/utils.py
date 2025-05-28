from json import dump
from jsonschema import validate
from os import makedirs 
from os.path import join
from rl.stateless.types import CheckpointConfigAlias, RoutesStatsAlias

# Required schema for checkpoint configs.
checkpoints_schema = {
    "type": "object",
    "properties": {
        "n": {"type": "number"},
        "output_path": {"type": "string"},
    },
    "required": ["n", "output_path"],
}

def validate_checkpoints_config(checkpoint_config:CheckpointConfigAlias)->bool:
    f'''
        Validates a provided checkpoint configuration for an stateless simulation.
        
        Parameters
        ----------
        `checkpoint_config`: {CheckpointConfigAlias} 
            A dictionary declaring the desired configs.

        Returns
        -------
        `True` if the dictionary schema is correct or no configuration is furnished. 

        Raises
        ------
        `ValueError` if the schema is wrong.
    '''
    if checkpoint_config:
        try:
            validate(instance=checkpoint_config, schema=checkpoints_schema)
            return True
        except:
            raise('`checkpoints` argument provided with invalid schema.')
    else:
        return True

def save_checkpoint(routes_stats:RoutesStatsAlias, n:int, output_dir:str)->None:
    f'''
        Saves the routes' data for a given iteration. 

        Parameters
        ----------
        `routes_stats`: {RoutesStatsAlias}
            The dictionary pointing the routes methods to their respective usage stats.
        `n`: int
            The iteration number.
        `output_dir`: str
            The directory to save the checkpoint.
    '''
    data = {
        'n_checkpoint':n,
        'stats':{
            str(route):routes_stats[route] for route in routes_stats
        }
    }
    makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, f'checkpoint_{n}.json'), 'w') as file:
        dump(data, file, ensure_ascii=False)

def checkpoint_policy(checkpoint_config:CheckpointConfigAlias, routes_stats:RoutesStatsAlias, n:int)->None:
    f'''
        Saves the checkpoint of a given iteration if it is according to the imposed policy.

        Parameters
        ----------
       `checkpoint_config`: {CheckpointConfigAlias} 
            The checkpoint configuration dictionary.
        `routes_stats`: {RoutesStatsAlias} 
            The dictionary pointing the routes methods to their respective usage stats.
        `n`: int
            The iteration number.
    '''
    if checkpoint_config:
        if n % checkpoint_config['n'] == 0:
            save_checkpoint(routes_stats, n, checkpoint_config['output_path'])