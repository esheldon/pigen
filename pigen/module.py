from . import config
from . import funcwrap
from .version import __version__

def make_wrapper(conf, output_file):
    """
    create a python wrapper based on the input description
    file

    parameters
    ----------
    conf: dict or yaml file name
        Description file
    output_file: string
        path to the output file
    """
    module=ModuleWrapper(conf)
    module.write(output_file)


class ModuleWrapper(dict):
    """
    module wrapper

    wrap the functions and create the python module C extension
    """
    def __init__(self, conf):
        self.conf = config.load_config(conf)

        self['modulename'] = self.conf['modulename']
        self['pigen_version'] = __version__

        self._set_func_wraps()
        self._set_includes()
        self._set_text()

    def write(self, fname):
        """
        write the module to the specified file
        """
        with open(fname,'w') as fobj:
            fobj.write(self._text)

    def get_text(self):
        """
        get the full text for the C extension module
        """
        return self._text

    def _set_text(self):
        """
        Set the full text
        """
        self._text = _template % self

    def _set_includes(self):
        """
        set the includes if any
        """
        if 'includes' in self.conf:
            self['includes'] = '\n'.join(self.conf['includes'])
        else:
            self['includes'] = ''

    def _set_func_wraps(self):
        """
        get all the function wrappers and the py method
        definitions
        """
        funcwraps=[]
        for d in self.conf['functions']:
            fwrap = funcwrap.FunctionWrapper(
                funcdef=d,
                prefix=self.conf['wrapper_prefix'],
                prefix_var_names=self.conf['prefix_var_names'],
            )
            funcwraps.append(fwrap)

        texts = [f.get_text() for f in funcwraps]
        self['all_function_texts'] = '\n\n'.join(texts)

        pydefs = ['    '+f.get_py_method_def() for f in funcwraps]

        pydefs += ['{NULL}']
        self['py_method_defs'] = ',\n'.join(pydefs)


    def __repr__(self):
        return self.get_text()

_template="""/*
    This module automatically generated by pigen %(pigen_version)s
    https://github.com/esheldon/pigen/
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include "numpy/arrayobject.h" 

%(includes)s

%(all_function_texts)s

static PyMethodDef %(modulename)s_py_method_defs [] = {

%(py_method_defs)s

};


#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "%(modulename)s",      /* m_name */
        "Defines the funcs associated with module",  /* m_doc */
        -1,                  /* m_size */
        %(modulename)s_py_method_defs, /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_%(modulename)s(void)
#else
init%(modulename)s(void) 
#endif
{
    PyObject* m=NULL;

#if PY_MAJOR_VERSION >= 3

    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else

    m = Py_InitModule3("%(modulename)s", %(modulename)s_py_method_defs, "Define funcs.");
    if (m==NULL) {
        return;
    }
#endif

    // for numpy
    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif

}
"""
