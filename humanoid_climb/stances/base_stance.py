from typing import List, Union


class Stance():
    def __init__(self, previous_stance: Union[List[int], None], state_file: Union[str, None], stance: List[int], action_override: List[int], exclude_targets: List[List[int]]):
        self.root_path = None
        self.previous_stance = previous_stance
        self.state_file = state_file
        self.stance = stance
        self.action_override = action_override
        self.exclude_targets = exclude_targets

    def get_args(self):
        state_path = self.root_path+self.state_file if self.state_file is not None else None
        dict = {'motion_path': [self.stance],
                'state_file': state_path,
                'action_override': self.action_override,
                'motion_exclude_targets': [self.exclude_targets]}
        return dict
