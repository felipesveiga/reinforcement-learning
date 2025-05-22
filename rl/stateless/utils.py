from json import dump
from jsonschema import validate
from os import makedirs 
from os.path import join
from rl.stateless.types import CheckpointConfigAlias, RoutesStatsAlias

checkpoints_schema = {
    "type": "object",
    "properties": {
        "n": {"type": "number"},
        "output_path": {"type": "string"},
    },
    "required": ["n", "output_path"],
}

def validate_checkpoints_config(checkpoint_config:CheckpointConfigAlias):
    if checkpoint_config:
        try:
            validate(instance=checkpoint_config, schema=checkpoints_schema)
            return True
        except:
            raise('`checkpoints` argument provided with invalid schema.')
    else:
        return True

def post_checkpoint(routes_stats:RoutesStatsAlias, i:int, output_dir:str)->None:
    data = {
        'n_checkpoint':i,
        'stats':{
            str(route):routes_stats[route] for route in routes_stats
        }
    }
    makedirs(output_dir, exist_ok=True)
    dump(data, join(output_dir, f'data_iteration_{i}.json'), ensure_ascii=False)

def checkpoint_policy(checkpoint_config:CheckpointConfigAlias, routes_stats:RoutesStatsAlias, i:int)->None:
    if checkpoint_config:
        if i % checkpoint_config['n'] == 0:
            post_checkpoint(routes_stats, i, checkpoint_config['output_path'])