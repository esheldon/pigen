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

def get_wrap_type(type):
    """
    get a type for wrapping.
    
    E.g. if the user has type int, this will be wrapped by int. But for size_t
    there is no direct mapping, so we choose unsigned long long, etc.

    This is a way to map types to something that python knows about.  Since we
    demand these are passed by value to the underlying function, we expect the
    conversion to be ok
    """
    if type not in _wrap_type_map:
        raise ValueError("don't know how to wrap type: '%s'" % type)
    return _wrap_type_map[type]


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


def get_py_format_code(type):
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
