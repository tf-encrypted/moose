pymoose package
===============

Subpackages
-----------

.. toctree::
   :maxdepth: 2

   pymoose.computation
   pymoose.edsl
   pymoose.predictors
   pymoose.logger
   pymoose.runtime

.. currentmodule:: pymoose

eDSL Values & Types
--------------------

.. autosummary::
   :toctree: _autosummary

   Argument
   AesKeyType
   AesTensorType
   FloatType
   IntType
   TensorType
   StringType

eDSL DTypes
---------------
.. autosummary::
   :toctree: _autosummary
   
   bool_
   fixed
   float32
   float64
   int32
   int64
   uint64
   ring64

eDSL Placement Factories
-------------------------

.. autosummary::
   :toctree: _autosummary

   host_placement
   mirrored_placement
   replicated_placement

eDSL Functions
---------------

.. autosummary::
   :toctree: _autosummary
   
   abs
   add
   add_n
   argmax
   atleast_2d
   cast
   concatenate
   constant
   decrypt
   div
   dot
   exp
   expand_dims
   greater
   identity
   index_axis
   inverse
   less
   load
   log
   log2
   logical_and
   logical_or
   maximum
   mean
   mul
   mux
   ones
   output
   relu
   reshape
   save
   select
   shape
   sliced
   softmax
   square
   squeeze
   strided_slice
   sigmoid
   sub
   sum
   sqrt
   transpose
   zeros

PyMoose Context
----------------

.. autosummary::
   :toctree: _autosummary

   get_current_placement
   get_current_runtime
   set_current_runtime

Moose Bindings
---------------

.. autosummary::
   :toctree: _autosummary
   
   ~pymoose.edsl.base.computation
   elk_compiler
   GrpcMooseRuntime
   LocalMooseRuntime
   MooseComputation
