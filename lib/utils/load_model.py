import mxnet as mx


def load_checkpoint(prefix, epoch, argprefix=''):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            if name[:len(argprefix)] == argprefix:
                arg_params[name] = v
            else:
                arg_params[argprefix + name] = v
        if tp == 'aux':
            if name[:len(argprefix)] == argprefix:
                aux_params[name] = v
            else:
                aux_params[argprefix + name] = v
    return arg_params, aux_params

def load_checkpoint_multi(prefix1, prefix2, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict_1 = mx.nd.load('%s-%04d.params' % (prefix1, epoch))
    save_dict_2 = mx.nd.load('%s-%04d.params' % (prefix2, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict_1.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    for k, v in save_dict_2.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

def convert_context(params, ctx):
    """
    :param params: dict of str to NDArray
    :param ctx: the context to convert to
    :return: dict of str of NDArray with context ctx
    """
    new_params = dict()
    for k, v in params.items():
        new_params[k] = v.as_in_context(ctx)
    return new_params


def load_param(prefix, epoch, convert=False, ctx=None, process=False, argprefix=''):
    """
    wrapper for load checkpoint
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :param convert: reference model should be converted to GPU NDArray first
    :param ctx: if convert then ctx must be designated.
    :param process: model should drop any test
    :return: (arg_params, aux_params)
    """
    arg_params, aux_params = load_checkpoint(prefix, epoch, argprefix)
    if convert:
        if ctx is None:
            ctx = mx.cpu()
        arg_params = convert_context(arg_params, ctx)
        aux_params = convert_context(aux_params, ctx)
    if process:
        tests = [k for k in arg_params.keys() if '_test' in k]
        for test in tests:
            arg_params[test.replace('_test', '')] = arg_params.pop(test)
    return arg_params, aux_params

def load_param_multi(prefix1, prefix2, epoch, convert=False, ctx=None, process=False):
    """
    wrapper for load checkpoint
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :param convert: reference model should be converted to GPU NDArray first
    :param ctx: if convert then ctx must be designated.
    :param process: model should drop any test
    :return: (arg_params, aux_params)
    """
    arg_params, aux_params = load_checkpoint_multi(prefix1, prefix2, epoch)
    if convert:
        if ctx is None:
            ctx = mx.cpu()
        arg_params = convert_context(arg_params, ctx)
        aux_params = convert_context(aux_params, ctx)
    if process:
        tests = [k for k in arg_params.keys() if '_test' in k]
        for test in tests:
            arg_params[test.replace('_test', '')] = arg_params.pop(test)
    return arg_params, aux_params
