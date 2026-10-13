"""Use the corrected fixed baseline's environment/evaluator unchanged."""
import argparse
import json
from pathlib import Path
from chainerrl import misc
from fd_env import FDEnvSelHeur
from evaluate import eval_performance
from policies import SimpleAgent, POLICIES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', choices=POLICIES, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    misc.set_random_seed(args.seed)
    env = FDEnvSelHeur(host='127.0.0.1', port=args.port, port_file_id=args.port,
        config_dir='.', num_heuristics=4, use_general_state_info=False,
        state_type=1, max_rand_steps=0, time_step_limit=2**63-1, temporal_window=0)
    env.seed(args.seed)
    try:
        stats = eval_performance(env, SimpleAgent(args.policy), None, 1,
                                 max_episode_len=2**63-1)
        Path('eval.json').write_text(json.dumps(stats))
    finally:
        env.close()


if __name__ == '__main__':
    main()
