Xarray Integration
==================

.. automodule:: jlnn.utils.xarray_utils
   :members:
   :undoc-members:

This module allows exporting model outputs to the ``xarray.Dataset`` format. This brings advantages such as:
* **Labelled dimensions**: Access to data via predicate names instead of numerical indices.
* **Advanced indexing**: Easy filtering of samples that exhibit high uncertainty.
* **Serialization**: Easy saving of experimental results in NetCDF format.
