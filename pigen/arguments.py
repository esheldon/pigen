from . import util

class Argument(dict):
    """
    Get information for wrapping a function argument
    """
    def __init__(self, argdef, prefix=None):
        self._argdef=argdef
        self['prefix'] = prefix
        self['type'], name = util.get_type_and_name(argdef)

        if self['prefix'] is not None:
            self['name'] = '%s_%s' % (prefix, name)
        else:
            self['name'] = name

        self._set_wrapper_info()
        self._set_unwrap_code()

    def _set_wrapper_info(self):
        """
        set the basic wrapper info
        """
        if 'PyObject' in self['type']:

            self['parse_tuple_argtype'] = 'PyObject*'

            self['wrap_name'] = self['name']
            self['wrapper_type'] = 'PyObject*'

            self['py_format'] = 'O'
            self._do_unwrap=False

        elif '*' in self['type'] and 'char' not in self['type']:
            self['parse_tuple_argtype'] = 'PyObject*'

            self['wrap_name'] = '%s_wrap' % self['name']
            self['wrapper_type'] = self['type']

            self['py_format'] = 'O'
            self._do_unwrap=True

        else:
            # a scalar
            self['parse_tuple_argtype'] = util.get_wrap_type(self['type'])

            self['wrap_name'] = self['name']
            self['wrapper_type'] = self['parse_tuple_argtype']

            self['py_format'] = util.get_py_format_code(self['wrapper_type'])
            self._do_unwrap=False

        self['parse_tuple_arg'] = '&%s' % self['wrap_name']


        tup=(self['wrapper_type'],self['name'])
        self['declaration'] = '    %s %s;' % tup

        if self._do_unwrap:
            tup=(self['parse_tuple_argtype'], self['wrap_name'])
            self['wrap_declaration'] = '    %s %s;' % tup

        else:
            self['wrap_declaration'] = None

    def _set_unwrap_code(self):
        """
        set the unwrap code, if needed
        """
        if self._do_unwrap:
            c = _array_unwrap_template % self
        else:
            c=None

        self['unwrap_code'] = c

_array_unwrap_template="""    if ( !PyArray_Check(%(wrap_name)s) ) {
        PyErr_SetString(PyExc_TypeError, "argument %(name)s must be an array");
        return NULL;
    }
    %(name)s = (%(type)s) PyArray_DATA( (PyArrayObject*) %(wrap_name)s );
"""



class Arguments(dict):
    """
    information for wrapping function arguments
    """
    def __init__(self, arglist, prefix=None):
        self._arglist=arglist
        self['prefix']=prefix

        self._args = [Argument(a,prefix) for a in arglist]

        self._set_py_format_string()
        self._set_unwraps()
        self._set_wrap_declarations()
        self._set_declarations()
        self._set_parse_tuple_args()
        self._set_function_args()

    def _set_py_format_string(self):
        """
        combined python type formats
        """
        slist = [a['py_format'] for a in self._args]
        self['py_formats'] = ''.join(slist)

    def _set_unwraps(self):
        """
        combined unwrap code
        """
        uwlist=[a['unwrap_code'] for a in self._args if a['unwrap_code'] is not None]
        if len(uwlist) == 0:
            self['unwraps']=None
        else:
            self['unwraps'] = '\n'.join(uwlist)

    def _set_wrap_declarations(self):
        """
        combined wrap declarations
        """
        dlist=[a['wrap_declaration'] for a in self._args
               if a['wrap_declaration'] is not None]

        if len(dlist) > 0:
            self['wrap_declarations'] = '\n'.join(dlist)
        else:
            self['wrap_declarations'] = None

    def _set_declarations(self):
        """
        combined wrap declarations
        """
        dlist=[a['declaration'] for a in self._args
               if a['declaration'] is not None]
        if len(dlist) == 0:
            self['declarations'] = None
        else:
            self['declarations'] = '\n'.join(dlist)


    def _set_parse_tuple_args(self):
        """
        the combined args for PyArg_ParseTuple
        """
        args=[a['parse_tuple_arg'] for a in self._args]
        self['parse_tuple_args']=', '.join(args)

    def _set_function_args(self):
        """
        the combined args to the wrapped function
        """
        args=[a['name'] for a in self._args]
        self['function_args'] = ', '.join(args)


