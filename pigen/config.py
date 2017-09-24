_config_defaults = dict(
    prefix='Py',
    prefix_var_names=False,
)

def load_config(fname):
    """
    load a config, setting default parameters

    parameters
    ----------
    fname: string or dict
        Either a dict representing a config or a filename

        In either case set default values
    """
    if isinstance(fname,dict):
        data=fname
    else:
        data=load_yaml(fname)

    conf={}
    conf.update(_config_defaults)
    conf.update(data)

    if 'wrapper_prefix' not in conf:
        conf['wrapper_prefix']='%s%s' % \
                (conf['prefix'], conf['modulename'])

    return conf

def load_yaml(fname):
    """
    load some data from a yaml file
    """
    import yaml
    with open(fname) as fobj:
        data=yaml.load(fobj)
    return data


