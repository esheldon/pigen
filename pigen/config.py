_config_defaults = dict(
    prefix='Py',
    prefix_var_names=False,
)

def load_yaml(fname):
    import yaml
    with open(fname) as fobj:
        data=yaml.load(fobj)
    return data

def load_config(fname):

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

