Visualization Tools
===================

.. automodule:: jlnn.utils.visualize
   :members:
   :undoc-members:

Graphical tools for visual audit of the model. JLNN emphasizes that users should see not only the result, but also the "space of doubt".

Visualization of Intervals
--------------------------
Function ``plot_truth_intervals`` draws horizontal graphs where consistent states (blue) are distinguished from logical contradictions (red).

Analysis of Weights
-------------------
Function ``plot_gate_weights`` allows to visualize the importance of individual inputs for a specific logical decision (e.g., which sensors most influence the rule for an alarm).

.. autofunction:: plot_truth_intervals
.. autofunction:: plot_gate_weights
.. autofunction:: plot_training_log_loss