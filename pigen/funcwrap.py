from __future__ import print_function, absolute_import

class FuncWrapper(dict):
    """
    function definition wrapper

    date a C function declaration and generate the python C 
    api wrapper code for it
    """
    def __init__(self, funcdef, prefix):
        self._prefix=prefix
        self._funcdef=funcdef

        self._set_defs()
        self._set_wrapper_funcdef()
        self._set_parse_tuple_call()
        self._set_function_call()

        self._combine()

    def get_text(self):
        """
        Get the text of the wrapper function definition
        """
        return self._text

    def _combine(self):
        """
        combine all the pieces into the wrapper function definition
        """
        args=self._args
        funcdef = self._funcdef

        text_list=[
            self._wrapper_funcdef,
            '{',
        ]

        if funcdef['return_def'] is not None:
            text_list += ['',funcdef['return_def']]

        if args is not None:
            if args['wrap_declarations'] is not None:
                text_list += ['',args['wrap_declarations']]

        if self._parse_tuple_call is not None:
            text_list += [
                '',
                self._parse_tuple_call,
            ]

        if args is not None and args['unwraps'] is not None:
            text_list += [
                '',
                args['unwraps'],
            ]
        text_list += [
            '',
            self._function_call,
            '',
            funcdef['return_call'],
            '',
            '}',
        ]


        self._text='\n'.join(text_list)

    def _set_defs(self):
        """
        set the argument defs, return type def, function defs
        """
        fs=self._funcdef.replace(';','').replace(')','')

        front,back = fs.split('(')
        self._funcdef = FuncDef(front)

        back=back.strip()
        if back=='void' or back=='':
            self._args=None
        else:

            arglist=[a.strip() for a in back.split(',')]
            self._args = Arguments(arglist)

    def _set_parse_tuple_call(self):
        """
        set the PyArg_ParseTuple call if needed
        """
        if self._args==None:
            self._parse_tuple_call=None
        else:
            self._parse_tuple_call = _parse_tuple_template % self._args

    def _set_wrapper_funcdef(self):
        """
        Set the wrapper function definition
        """
        self._wrapper_funcdef = _wrapper_funcdef_template % dict(
            prefix=self._prefix,
            func_name=self._funcdef['func_name'],
        )

    def _set_function_call(self):
        """
        set the call to the wrapped function
        """
        if self._args is None:
            fargs=''
        else:
            fargs=self._args['function_args']
        fcall='%s( %s );' % (
            self._funcdef['func_name'],
            fargs,
        )

        rname=self._funcdef['return_var_name']
        if rname is not None:
            fcall = '%s = %s' % (rname, fcall)

        self._function_call='    %s' % fcall

           
    def __repr__(self):
        return self._text

class FuncDef(dict):
    """
    extract information about the wrapped function
    """
    def __init__(self, front):
        if '*' in front and 'PyObject' not in front:
            raise RuntimeError("pointer return not supported, except PyObject*")

        self['return_type'], self['func_name'] = get_type_and_name(front)

        self._set_return_var()
        self._set_return_call()

    def _set_return_var(self):
        """
        set up info for the return variable if it exists
        """
        if 'void' in self['return_type']:
            self['return_var_name']=None
            self['return_def']=None
        else:
            self['return_var_name']='%s_retval' % self['func_name']
            self['return_def'] = '    %s %s;' % \
                    (self['return_type'], self['return_var_name'])

    def _set_return_call(self):
        """
        set the return call
        """
        if self['return_type'] == 'void':
            self['return_call'] = '    Py_RETURN_NONE;'
        elif 'PyObject' in self['return_type']:
            self['return_call']='    return %s;' % self['return_var_name']
        else:
            pytype = get_pytype(self['return_type'])
            self['return_call']='    return Py_BuildValue("%s", %s);' % \
                    (pytype, self['return_var_name'])
 
class Argument(dict):
    """
    Get information for wrapping a function argument
    """
    def __init__(self, argdef):
        self._argdef=argdef
        self['type'], self['name'] = get_type_and_name(argdef)

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

            self['pytype'] = 'O'
            self._do_unwrap=False

        elif '*' in self['type']:
            self['parse_tuple_argtype'] = 'PyObject*'

            self['wrap_name'] = '%s_wrap' % self['name']
            self['wrapper_type'] = self['type']

            self['pytype'] = 'O'
            self._do_unwrap=True

        else:
            self['parse_tuple_argtype'] = get_wrap_type(self['type'])

            self['wrap_name'] = self['name']
            self['wrapper_type'] = self['parse_tuple_argtype']

            self['pytype'] = get_pytype(self['wrapper_type'])
            self._do_unwrap=False

        self['parse_tuple_arg'] = '&%s' % self['wrap_name']

        tup=(self['parse_tuple_argtype'], self['wrap_name'])
        self['wrap_declaration'] = '    %s %s;' % tup

    def _set_unwrap_code(self):
        """
        set the unwrap code, if needed
        """
        if self._do_unwrap:
            c='    %(name)s = (%(type)s) PyArray_DATA( (PyArrayObject*) %(wrap_name)s );'
            c = c % self
        else:
            c=None

        self['unwrap_code'] = c

class Arguments(dict):
    """
    information for wrapping function arguments
    """
    def __init__(self, arglist):
        self._arglist=arglist

        self._args = [Argument(a) for a in arglist]

        self._set_pytype_string()
        self._set_unwraps()
        self._set_wrap_declarations()
        self._set_parse_tuple_args()
        self._set_function_args()

    def _set_pytype_string(self):
        """
        combined python type formats
        """
        slist = [a['pytype'] for a in self._args]
        self['pytypes'] = ''.join(slist)

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
        dlist=[a['wrap_declaration'] for a in self._args]
        self['wrap_declarations'] = '\n'.join(dlist)

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


def get_type_and_name(d):
    """
    extract type and variable name

    parameters
    ----------
    cdef: string
        e.g.
          int x
          double *y
    
    returns:
        type, name
    """
    if '*' in d:
        split=d.split('*')
        type = split[0].strip() + '*'
        name = split[1].strip()
    else:
        split=d.split(' ')
        type = split[0].strip()
        name = split[1].strip()

    return type, name

def get_pytype(type):
    """
    get the python format code for conversions, e.g.
    used by PyBuild_Value
    """
    if '*' in type:
        return 'O'
    else:
        if type not in _pytype_map:
            raise ValueError("don't know parse tuple string for type: '%s'" % type)
        return _pytype_map[type]

def get_wrap_type(type):
    """
    get a type for wrapping.  E.g. if the user has type int, this will
    be wrapped by int. But for size_t there is no direct mapping, so
    we choose unsigned long long
    """
    if type not in _wrap_type_map:
        raise ValueError("don't know how to wrap type: '%s'" % type)
    return _wrap_type_map[type]




_wrapper_funcdef_template='PyObject* %(prefix)s_%(func_name)s(PyObject* self, PyObject* args)'


_parse_tuple_template="""
    if (!PyArg_ParseTuple(args, (char*)"%(pytypes)s", 
                          %(parse_tuple_args)s)) {
        return NULL;
    }"""

# this is a way to map types to something that python
# knows about.  Since we demand these are passed by
# value to the underlying function, we expect the
# conversion to be ok

_wrap_type_map={
    'double':'double',
    'float':'float',

    'short':'short int',
    'short int ':'short int',
    'unsigned short':'unsigned short int',
    'unsigned short int ':'unsigned short int',

    'int':'int',
    'unsigned int':'insigned int',

    'long':'long',
    'long int':'long',
    'unsigned long':'unsigned long',
    'unsigned long int':'unsigned long',

    'long long':'long long',

    'ssize_t':'long long',
    'size_t':'unsigned long long',
}

# the codes for type conversions, e.g. for Py_BuildValue or
# for parse tuple
_pytype_map={
    'double':'d',
    'float':'d',

    'short':'h',
    'short int':'h',
    'unsigned short':'H',
    'unsigned short int':'H',

    'int':'i',
    'unsigned int':'I',

    'long':'l',
    'long int':'l',
    'unsigned long':'k',
    'unsigned long int':'k',

    'long long':'L',
    'unsigned long long':'K',
}    


def test():
    modulename='_gmix'
    prefix='Py%s' % modulename
    # prefix will be 'Py_%s' % modulename
    defs = [
        'void noarg_or_return(void)',
        'int noarg()',
        'float scalar(float x);',
        'double fdouble(double * y, size_t ny);',
        'void fill_fdiff(struct gauss* gmix, long n_gauss, double *fdiff, long n_fdiff);',
        'PyObject* fpyobj(PyObject* input1, PyObject* input2);',
    ]

    for d in defs:
        print('// ' + '-'*70)
        print()

        wrapper=FuncWrapper(d, prefix)
        print(wrapper)
        print()


if __name__=="__main__":
    test()
