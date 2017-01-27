
import tensorflow as tf

from algorithms.batch_policy.base import BatchPolicyBase


class TRPO(BatchPolicyBase):

    def __init__(self, *args, **kwargs):
        super(TRPO, self).__init__(*args, **kwargs)

    def _init_variables(self):
        pass

    def _optimize_policy(self):
        pass