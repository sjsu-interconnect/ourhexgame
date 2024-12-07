from g01agent_oskar import G01Agent as OskarAgent

from ourhexenv import OurHexGame

class G01Agent(OskarAgent):
    """
    We have this structure to allow for us to easily swap our whose model we wish to use based on the super class
    ( oskar or zi's implementation ) we inherit from.
    """
    def __init__(self, env: OurHexGame):
        if env.sparse_flag:
            model_weights = 'sparse_model.pt'
        else:
            model_weights = 'dense_model.pt'

        super().__init__(env.board_size, load_checkpoint=model_weights)

