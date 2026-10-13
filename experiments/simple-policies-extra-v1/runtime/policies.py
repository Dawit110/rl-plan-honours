"""Fixed historical settings; no training or test-set tuning."""
import math
from historical import (NormalizedProgressAgent, StagnationProgressAgent,
                        ProbeThenExploitAgent, ProgressQueueAgent)

POLICIES = ('recent50','stagprog50','probe25x100','queue50a1')


class SimpleAgent:
    def __init__(self, policy):
        common = dict(actions=[0,1,2,3],verbose=False)
        if policy == 'recent50':
            self.agent = NormalizedProgressAgent(window=50,initial_action=0,**common)
        elif policy == 'stagprog50':
            self.agent = StagnationProgressAgent(threshold=50,window=50,initial_action=0,**common)
        elif policy == 'probe25x100':
            self.agent = ProbeThenExploitAgent(probe_length=25,exploit_length=100,**common)
        elif policy == 'queue50a1':
            self.agent = ProgressQueueAgent(window=50,alpha=1,initial_action=0,**common)
        else:
            raise ValueError('Unknown historical policy: '+policy)

    def act(self, observation):
        if len(observation) != 20:
            raise ValueError('Expected 20 RAW queue features')
        queues=[float(observation[5*h+3]) for h in range(4)]
        if not all(math.isfinite(q) and q>=0 for q in queues):
            raise ValueError('Invalid queue size')
        if not any(q>0 for q in queues):
            # Historical probe loop otherwise retries empty probes forever.
            # Return control to the planner so it can report exhausted search.
            return 0
        if any(not math.isfinite(float(observation[5*h+2])) for h in range(4)):
            raise ValueError('Nonfinite queue minimum after environment processing')
        return self.agent.act(observation)

    def stop_episode(self):
        return self.agent.stop_episode()
