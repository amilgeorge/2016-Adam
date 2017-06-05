
POLICY_OFFSET = 'offset'
POLICY_DEFAULT_ZERO = 'default_zero'

def with_offset(offset):
    #assert offset > 0, "expected offset value > 0 "
    return lambda frame_no: max(0,frame_no - offset)

def default_zero():
    return lambda frame_no:0



def get(policy,arg0 = None):

    if policy==POLICY_OFFSET:
        return with_offset(arg0)
    elif policy==POLICY_DEFAULT_ZERO:
        assert (arg0 is None or arg0==0),'invalid arg0 for policy '
        return default_zero()
