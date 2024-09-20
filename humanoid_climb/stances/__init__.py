from humanoid_climb.stances.base_stance import Stance

_STANCES_ = []

STANCE_NONE = Stance(stance=[-1, -1, -1, -1],
                     previous_stance=None,
                     state_file=None,
                     action_override=[-1, -1, -1, -1],
                     exclude_targets=[[], [], [], []])

STANCE_1 = Stance(stance=[10, 9, -1, -1],
                  previous_stance=None,
                  state_file=None,
                  action_override=[-1, -1, -1, -1],
                  exclude_targets=[[], [], [], []])

STANCE_2 = Stance(stance=[10, 9, 2, -1],
                  previous_stance=[10, 9, -1, -1],
                  state_file="/states/state_10_9_n_n.npz",
                  action_override=[1, 1, -1, -1],
                  exclude_targets=[[], [], [], []])

STANCE_3 = Stance(stance=[10, 9, 2, 1],
                  previous_stance=[10, 9, 2, -1],
                  state_file="/states/state_10_9_2_n.npz",
                  action_override=[1, 1, 1, -1],
                  exclude_targets=[[], [], [], []])

STANCE_4 = Stance(stance=[10, 13, 2, 1],
                  previous_stance=[10, 9, 2, 1],
                  state_file="/states/state_10_9_2_1.npz",
                  action_override=[1, -1, 1, 1],
                  exclude_targets=[[], [9], [], []])

STANCE_5 = Stance(stance=[10, 13, 2, 5],
                  previous_stance=[10, 13, 2, 1],
                  state_file="/states/state_10_13_2_1.npz",
                  action_override=[1, 1, 1, -1],
                  exclude_targets=[[], [], [], [1]])

STANCE_6 = Stance(stance=[13, 13, -1, 5],
                  previous_stance=[10, 13, 2, 5],
                  state_file="/states/state_10_13_2_5.npz",
                  action_override=[-1, 1, -1, 1],
                  exclude_targets=[[10], [], [1], []])

STANCE_7 = Stance(stance=[13, 13, 6, 5],
                  previous_stance=[13, 13, -1, 5],
                  state_file="/states/state_13_13_n_5.npz",
                  action_override=[1, 1, -1, 1],
                  exclude_targets=[[], [], [1, 2], []])

# Similar to Stance3
STANCE_8 = Stance(stance=[14, 13, 6, 5],
                  previous_stance=[13, 13, 6, 5],
                  state_file="/states/state_13_13_6_5.npz",
                  action_override=[-1, 1, 1, 1],
                  exclude_targets=[[13, 10], [], [], []])

STANCE_9 = Stance(stance=[14, 17, 6, 5],
                  previous_stance=[14, 13, 6, 5],
                  state_file="/states/state_14_13_6_5.npz",
                  action_override=[1, -1, 1, 1],
                  exclude_targets=[[], [13, 9, 12, 14], [], []])

STANCE_10 = Stance(stance=[14, 17, -1, 9],
                   previous_stance=[14, 17, 6, 5],
                   state_file="/states/state_14_17_6_5.npz",
                   action_override=[1, 1, -1, -1],
                   exclude_targets=[[], [], [6, 5], [4, 5, 8]])

STANCE_11 = Stance(stance=[-1, 17, 10, 9],
                   previous_stance=[14, 17, -1, 9],
                   state_file="/states/state_14_17_n_9.npz",
                   action_override=[-1, 1, -1, 1],
                   exclude_targets=[[], [], [6, 5, 11], []])

STANCE_11_2 = Stance(stance=[18, 17, 10, 9],
                     previous_stance=[14, 17, -1, 9],
                     state_file="/states/state_14_17_n_9.npz",
                     action_override=[-1, 1, -1, 1],
                     exclude_targets=[[14, 10], [], [6, 5, 11], []])

STANCE_11_3 = Stance(stance=[14, 17, 10, 9],
                     previous_stance=[14, 17, -1, 9],
                     state_file="/states/state_14_17_n_9.npz",
                     action_override=[1, 1, -1, 1],
                     exclude_targets=[[], [], [6, 5, 11], []])

STANCE_12 = Stance(stance=[18, 17, 10, 9],
                   previous_stance=[14, 17, 10, 9],
                   state_file="/states/state_14_17_10_9.npz",
                   action_override=[-1, 1, 1, 1],
                   exclude_targets=[[14, 17, 10], [], [], []])

STANCE_13 = Stance(stance=[18, 22, -1, -1],
                   previous_stance=[18, 17, 10, 9],
                   state_file="/states/state_18_17_10_9.npz",
                   action_override=[1, -1, -1, -1],
                   exclude_targets=[[], [17, 21, 13, 9, 12, 10], [], [9, 6]])

STANCE_13_2 = Stance(stance=[18, 20, 10, 9],
                     previous_stance=[18, 17, 10, 9],
                     state_file="/states/state_18_17_10_9.npz",
                     action_override=[1, -1, 1, 1],
                     exclude_targets=[[], [17, 12, 13, 14], [], []])

STANCE_14 = Stance(stance=[20, 20, 10, -1],
                   previous_stance=[18, 20, 10, 9],
                   state_file="/states/state_18_20_10_9.npz",
                   action_override=[-1, 1, 1, -1],
                   exclude_targets=[[18, 14, 19], [], [], []])

STANCE_14_1 = Stance(stance=[20, 20, 10, 13],
                     previous_stance=[18, 20, 10, 9],
                     state_file="/states/state_18_20_10_9.npz",
                     action_override=[-1, 1, 1, -1],
                     exclude_targets=[[18, 14, 19], [], [], [9]])

_STANCES_.append(STANCE_NONE)
_STANCES_.append(STANCE_1)
_STANCES_.append(STANCE_2)
_STANCES_.append(STANCE_3)
_STANCES_.append(STANCE_4)
_STANCES_.append(STANCE_5)
_STANCES_.append(STANCE_6)
_STANCES_.append(STANCE_7)
_STANCES_.append(STANCE_8)
_STANCES_.append(STANCE_9)
_STANCES_.append(STANCE_10)
_STANCES_.append(STANCE_11)
_STANCES_.append(STANCE_11_2)
_STANCES_.append(STANCE_11_3)
_STANCES_.append(STANCE_12)
_STANCES_.append(STANCE_13)
_STANCES_.append(STANCE_13_2)
_STANCES_.append(STANCE_14)
_STANCES_.append(STANCE_14_1)


def set_root_path(root_path):
    for stance in _STANCES_:
        stance.root_path = root_path
