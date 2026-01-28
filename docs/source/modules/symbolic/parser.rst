Formula Parser
==============

.. automodule:: jlnn.symbolic.parser
   :members:
   :show-inheritance:

This module uses the **Lark** library to transform string definitions of rules into syntactic trees (CST).

LNN Grammar
-------------
Supports complex logical expressions including:
* **The weight of facts**: ``0.8::A`` (setting prior truth value).
* **Weighted rules**: ``A & B ->[0.9] C`` (setting the weight of implication).
* **Temporal operators**: ``G`` (Always), ``F`` (Eventually), ``X`` (Next).