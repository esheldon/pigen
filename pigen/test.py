from __future__ import print_function, absolute_import
from . import files
from . import funcwrap

def test(prefix_var_names=False):
    conf={
        'modulename': '_gmix',
        'prefix_var_names':prefix_var_names,

        'functions': [
            'void noarg_or_return(void)',
            'int noarg()',
            'float scalar(float x);',
            'double fdouble(double * y, size_t ny);',
            'void fill_fdiff(struct gauss* gmix, long n_gauss, double *fdiff, long n_fdiff);',
            'PyObject* fpyobj(PyObject* input1, PyObject* input2);',
        ]
    }

    conf=files.load_config(conf)

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


