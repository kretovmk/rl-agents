




class TRPOAgent(object):
    """
    Class for agent governed by TRPO.
    """
    def __init__(self, sess,
                       env,
                       policy,
                       state_processor,
                       gamma=0.99
                 ):
        self.sess = sess
        self.env = env
        self.policy = policy
        self.state_processor = state_processor
        self.gamma = gamma



