
import os

#----------------------------------------------------------------------------------------------

env_bb = os.environ.get('BB', '')
if env_bb == '1':
    try:
        from pudb import set_trace as bb
    except ImportError:
        try:
            from ipdb import set_trace as bb
        except ImportError:
            from pdb import set_trace as bb
elif env_bb == 'pudb':
    from pudb import set_trace as bb
elif env_bb == 'pdb':
    from pdb import set_trace as bb
elif env_bb == 'ipdb':
    from ipdb import set_trace as bb
else:
    def bb(): pass

#----------------------------------------------------------------------------------------------
