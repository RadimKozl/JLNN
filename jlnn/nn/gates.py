#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx
from jlnn.core import logic, intervals

class WeightedOr(nnx.Module):
    """
    Trainable weighted OR gate for LNN (Logical Neural Networks).

    This gate implements a weighted version of the Łukasiewicz disjunction (t-conorms). 
    Within the JLNN architecture, the weights and threshold (beta) are defined as optimizable parameters (nnx.Param), 
    allowing the logic gate to learn the importance of individual inputs directly from the data.

    The gate works with interval logic, 
    where the operation is performed independently of the lower (L) and upper (U) truth limits.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes the parameters of the weighted OR gate.

        Args:
            num_inputs (int): Number of input logical arguments (dimension of the input layer).
            rngs (nnx.Rngs): Random number generator required to initialize Flax NNX state.
        """
        # The weights are initialized to 1.0, which corresponds to the standard logical sum.
        # In LNN, it is recommended to keep weights >= 1 to maintain logical interpretability.
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        
        # Beta (threshold/bias) determines the sensitivity of the gate.
        # A value of 1.0 indicates standard Łukasiewicz semantics.
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a forward pass of the weighted disjunction.

        The calculation is performed according to the formula: f(x) = min(1, sum(w_i * x_i) / beta).

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2). 
                            The last dimension contains the pair [Lower Bound, Upper Bound].

        Returns:
            jnp.ndarray: Output truth interval of the form (..., 2).
        """
        # Delegating computation to a low-level implementation in core.logic
        return logic.weighted_or_lukasiewicz(x, self.weights, self.beta)


class WeightedAnd(nnx.Module):
    """
    Trainable weighted AND gate for LNN (Logical Neural Networks).
    
    This gate implements a weighted version of the Łukasiewicz conjunction (t-norm). 
    Within the JLNN architecture, it is used to aggregate conditions that must be met simultaneously. 
    By using nnx.Param, the weights and threshold (beta) are optimizable, 
    allowing the network to learn the relevance of individual inputs to the resulting conjunction.
    
    The gate directly operates on the truth intervals [L, U], thereby preserving information 
    about uncertainty (epistemic uncertainty) across the computational graph.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes the parameters of the weighted AND gate.

        Args:
            num_inputs (int): Number of input logical arguments (flags).
            rngs (nnx.Rngs): Random number generator for initializing the Flax NNX state.
        """
        
        # Weights are initialized to 1.0. A higher weight for a particular input
        # means that its (false) truth has a stronger influence on the result of the conjunction.
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        
        # Beta (threshold) determines the stringency of the gate. Higher beta dampens the influence of
        # negative evidence on the resulting truth.
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a forward calculation of the weighted conjunction.

        The calculation is carried out according to the formula: 1 - min(1, sum(w_i * (1 - x_i)) / beta).

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2). 
                            The last dimension contains the pair [Lower Bound, Upper Bound].

        Returns:
            jnp.ndarray: Output truth interval of the form (..., 2).
        """
        # Calling a low-level function from jlnn.core.logic for efficient computation
        return logic.weighted_and_lukasiewicz(x, self.weights, self.beta)


class WeightedNand(nnx.Module):
    """
    Trainable weighted NAND gate (Negated AND) for JLNN.
    
    In automation and industrial logic, NAND is one of the most versatile elements. 
    In JLNN, it is implemented as a composition of a weighted AND gate and a logical negation operation.
    
    Interval semantics:
    If an AND gate returns the interval [L, U], a NAND gate returns [1 - U, 1 - L].
    This means that the high rate of confirmed truth in the conjunction changes to 
    a high rate of confirmed falsehood in the NAND, 
    while the width of the interval (indeterminacy) remains correctly preserved.
    """    
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes the NAND gate using the internal WeightedAnd module.

        Args:
            num_inputs (int): Number of input logic signals.
            rngs (nnx.Rngs): Random number generator for parameter initialization.
        """
        # NAND uses an internal trainable AND gate that manages weights and beta.
        self.and_gate = WeightedAnd(num_inputs, rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a weighted NAND calculation over the input intervals.

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2).

        Returns:
            jnp.ndarray: The resulting truth interval [L, U] of the form (..., 2).
        """        
        # 1. Calculation of weighted conjunction: [L_and, U_and]
        res_and = self.and_gate(x)
        
        # 2. Applying negation: [1 - U_and, 1 - L_and]
        # Uses the intervals.negate function to ensure correct semantic mapping.
        return intervals.negate(res_and)


class WeightedNor(nnx.Module):
    """
    Trainable weighted NOR gate (Negated OR) for JLNN.
    
    In industrial automation, the NOR gate (Peirce operator) is indispensable for safety 
    and monitoring circuits. In the JLNN framework, 
    it is implemented as a composition of a weighted OR gate and a logical negation operation.
    
    Semantics in interval logic:
    The calculation is as NOT(WeightedOr(x)). 
    If the inner OR gate determines that there is at least one true input (high L), 
    the output of the NOR gate will have a high confirmed falseness rate (low U).
    
    This implementation preserves epistemic uncertainty: the width of the resulting interval 
    exactly corresponds to the degree of ignorance about the state of the input sensors.
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs):
        """
        Initializes a NOR gate using the internal WeightedOr module.

        Args:
            num_inputs (int): Number of input logic signals (e.g. sensors).
            rngs (nnx.Rngs): Random number generator for initializing internal parameters.
        """
        # NOR uses an internal trainable OR gate that manages weights and beta.
        # With Flax NNX, these parameters will be automatically found by the clip_weights function.        
        self.or_gate = WeightedOr(num_inputs, rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a weighted NOR calculation over the input intervals.

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., num_inputs, 2). 
                            The last dimension contains the pair [Lower Bound, Upper Bound].

        Returns:
            jnp.ndarray: The resulting truth interval [L, U] of the form (..., 2).
        """
        # 1. Calculation of weighted disjunction (Łukasiewicz t-conorm)        
        res_or = self.or_gate(x)
        # 2. Logická negace intervalu: [L, U] -> [1 - U, 1 - L]
        return intervals.negate(res_or)


class WeightedXor(nnx.Module):
    """
    Trainable weighted XOR gate (Exclusive OR) for JLNN.
    
    In automation, XOR is key for detecting mismatches (e.g. checking if two valves are in opposite states). 
    In JLNN it is implemented using the composition: (A OR B) AND (A NAND B).
    
    Thanks to the use of internal weighted gates, 
    this XOR can learn not only the classic logic table, 
    but also "fuzzy" transitions and the importance 
    of individual inputs for detecting exclusivity.
    
    The output interval correctly represents the degree of certainty 
    that the inputs are in a different logical state.
    """
    def __init__(self, rngs: nnx.Rngs):
        """
        Initializes the XOR gate using internal trainable modules.

        Args:
            rngs (nnx.Rngs): Random number generator for parameter initialization.
        """
                
        # XOR is defined for 2 inputs in this basic version.
        # We use previously defined gates, thus ensuring hierarchical
        # learning of weights and thresholds (beta) in the entire XOR subgraph.
        self.or_gate = WeightedOr(num_inputs=2, rngs=rngs)
        self.nand_gate = WeightedNand(num_inputs=2, rngs=rngs)
        
        # Final aggregation of results from OR and NAND branches.
        self.final_and = WeightedAnd(num_inputs=2, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs a forward weighted XOR calculation.

        Args:
            x (jnp.ndarray): Input interval tensor of the form (..., 2, 2). 
                            The penultimate dimension must contain exactly two logical intervals.
        Returns:
            jnp.ndarray: Output truth interval [L, U] of the form (..., 2).
        """
        
        # 1. Parallel calculation of disjunction and negated conjunction
        res_or = self.or_gate(x)
        res_nand = self.nand_gate(x)
        
        # 2. Combine the partial results into a new tensor for the final AND gate.
        # The stacking result has the form (..., 2, 2).
        combined = jnp.stack([res_or, res_nand], axis=-2)
        
        # 3. Final conjunction indicating exclusivity
        return self.final_and(combined)


class WeightedImplication(nnx.Module):
    """
    Trainable gate for logical implication (A -> B).
    
    This gate implements the relationship between antecedent (premise A) 
    and consequent (consequence B). Within JLNN, it supports multiple semantics, 
    allowing the user to choose between optimistic (Łukasiewicz), pessimistic (Kleene-Dienes), 
    or compromise (Reichenbach) approaches.
    
    The weights on the implication allow the model to learn the relevance 
    of individual parts of the rule, while the beta parameter controls 
    the strictness of the activation of the entire rule.
    """
    def __init__(self, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the parameters of the weighted implication.

        Args:
            rngs (nnx.Rngs): Random number generator for Flax NNX.
            method (str): Selected logical method. 
                        Supported values: 'lukasiewicz', 'kleene_dienes', 'reichenbach'.
        """
        self.method = method
        
        # An implication has 2 inputs: an antecedent (A) and a consequent (B).
        # The weights are initialized to 1.0, which corresponds to the standard logical strength.
        self.weights = nnx.Param(jnp.ones((2,)))
        # Beta determines the sensitivity threshold for activating the implication.
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the truth interval of an implication between two statements.

        Args:
            int_a (jnp.ndarray): Tensor for the antecedent (A) of the form (..., 2). 
            int_b (jnp.ndarray): Tensor for the consequent (B) of the form (..., 2).

        Returns:
            jnp.ndarray: The resulting truth interval [L, U] of the form (..., 2).
        
        Raises:
            ValueError: If an unsupported calculation method is set.
        """
        if self.method == 'lukasiewicz':
            # Łukasiewicz uses internal weighted OR (¬A ∨ B).
            return logic.implies_lukasiewicz(int_a, int_b, self.weights, self.beta)
        
        # For the Kleene-Dienes and Reichenbach methods, we apply weights as preprocessing.
        # This scales the importance of the input before applying the logical function itself.
        weighted_a = intervals.create_interval(
            jnp.minimum(1.0, intervals.get_lower(int_a) * self.weights[0]),
            jnp.minimum(1.0, intervals.get_upper(int_a) * self.weights[0])
        )
        weighted_b = intervals.create_interval(
            jnp.minimum(1.0, intervals.get_lower(int_b) * self.weights[1]),
            jnp.minimum(1.0, intervals.get_upper(int_b) * self.weights[1])
        )

        if self.method == 'kleene_dienes':
            # max(1 - A, B)
            return logic.implies_kleene_dienes(weighted_a, weighted_b)
        elif self.method == 'reichenbach':
            # 1 - A + (A * B)
            return logic.implies_reichenbach(weighted_a, weighted_b)
        else:
            raise ValueError(f"Unsupported implication method: {self.method}")