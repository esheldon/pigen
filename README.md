# pigen
Extremely simple python interface generator for C code

Motivation
----------

Provide a very simple way to interface with C code.  The goal is to make it
easy to use C for performance boosts,  while keeping this code, and the wrapped
C code, simple and easy to understand.  It is not a goal to wrap existing code
of all kinds, but rather code crafted to be easy to wrap.

Principles
----------

Some principles are followed in order to keep this code simple.

- Each function wrapper is only that: a wrapper.  Its purpose is to provide
  inputs to the wrapped function.
- Memory allocation should happen in python, before calling the C function.
  Allocate arrays and pass them in.
- Array type checking should happen in python, before calling the C function.
- Return values can only be None (void) or scalars, which can be easily
  returned using Py_BuildValue.  This follows from above: If the return value
  were a pointer, it would imply some kind of memory allocation, and returning it
  would involve counting references.  If multiple return values are needed, send
  some kind of array and fill the existing memory.  The exception is if the user
  declares the return type as PyObject*, then it is up the user to provide the
  return value.

Implementation Restrictions
---------------------------

Some restrictions may be lifted if an easy way to implement a particular case
is found.

- No memory allocation is allowed in the wrapper except by the use
  of Py_BuildValue for scalar return values.
- Scalar inputs are translated directly to c types. This includes
  strings, but only as const char *.
- Pointer inputs are assumed to represent numpy arrays, and the
  underlying pointer is extracted using PyArray_DATA.  It is
  checked that the input is a numpy array.  No array type or shape
  checking is performed.
- String arrays are not provided translation because they are ambiguous
  with ordinary strings.  If string arrays are needed, have your
  function take a PyObject* as an argument and do the PyArray
  calls yourself

Example definition file
-----------------------

```yaml
modulename: '_gmix'

includes:
  - gmix.h
  - point.h

# function defs are either a string with the prototype, or a dict with the
# prototype in the 'def' entry, and possibly a 'doc' entry for documentation.
# semicolons are ignored in prototypes

functions:
  - def: void noargs_or_return(void)
    doc: no args or return value

  - def: int noargs()
    doc: no args, int return

  - def: double dsum(double * y, size_t ny)
    doc: func returns double, takes array and array size


  # docs are optional
  - def: double dscalar(void)


  # if there is no doc, a simple string can be used
  - float fscalar(float x);


  # multiline
  - |
      void fill_fdiff(struct gauss* gmix,
                      long n_gauss,
                      double *fdiff,
                      long n)

  # multiline within dict
  - def: |
      PyObject* fpyobj(PyObject* input1,
                       PyObject* input2)
    doc: |
      All types are declared as PyObjects*. The user is responsible
      for using the python api.
```
TODO
-------
- support strings char * using const char * as the type
