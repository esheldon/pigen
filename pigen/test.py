from __future__ import print_function, absolute_import
from . import config
from . import module

def test(prefix_var_names=False):
    conf={
        'modulename': '_gmix',
        'prefix_var_names':prefix_var_names,

        'includes':[
            '#include "gmix.h"',
            '#include "fitting.h"',
        ],

        'functions': [
            {'def':'void noarg_or_return(void)',
             'doc':'no argument or return value'},
            'int noarg()',
            'float scalar(float x);',
            'double fdouble(double * y, size_t ny);',
            {'def':'void fill_fdiff(struct gauss* gmix, long n_gauss, double *fdiff, long n_fdiff);',
             'doc':'unpack user define struct array and double array'},
            {'def':'PyObject* fpyobj(PyObject* input1, PyObject* input2);',
             'doc':'All PyObject*, so the user must deal with extracting data and generating the PyObject* return value'},
        ]
    }

    conf=config.load_config(conf)

    mwrap = module.ModuleWrapper(conf)
    print(mwrap)

    '''
    for d in conf['functions']:
        print('// ' + '-'*70)
        print()

        wrapper=funcwrap.FuncWrapper(
            funcdef=d,
            prefix=conf['wrapper_prefix'],
            prefix_var_names=conf['prefix_var_names'],
        )
        print(wrapper)
        print()
    '''

