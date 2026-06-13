#!/usr/bin/env python3

# Imports
import jax.numpy as jnp
from flax import nnx
from jlnn.nn import functional as F

# =====================================================================
# 1. TRADITIONAL PARAMETRIC LOGIC GATES (Learning Weights and Beta)
# =====================================================================

class WeightedOr(nnx.Module):
    """
    Trainable parametric fuzzy OR gate supporting standard t-conorms.

    This module aggregates multiple input interval streams into a single truth
    interval using a parameterized t-conorm operation. It optimizes individual
    input importance via learnable weights and adjusts the global activation threshold
    using a learnable bias parameter (beta).

    Attributes:
        method (str): Target logical framework selector ('lukasiewicz', 'godel', 
            'kleene_dienes', 'product', 'reichenbach').
        weights (nnx.Param): Trainable input importance weights structured as (num_inputs,).
        beta (nnx.Param): Trainable gate activation sensitivity threshold scalar (bias).
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the stateful WeightedOr gate with corresponding optimization parameters.

        Args:
            num_inputs (int): Number of input streams to aggregate.
            rngs (nnx.Rngs): Flax NNX random number generator collection.
            method (str, optional): Fuzzy logic framework to employ. Defaults to 'lukasiewicz'.
        """
        self.method = method
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the parameterized forward pass for the fuzzy OR operation.

        Args:
            x (jnp.ndarray): Input interval tensor structured as (..., num_inputs, 2),
                where the final dimension contains the lower and upper bounds [L, U].

        Returns:
            jnp.ndarray: Evaluated collective truth interval structured as (..., 2).

        Raises:
            ValueError: If the specified `method` is not supported by the parametric OR layer.
        """
        if self.method == 'lukasiewicz':
            return F.weighted_or(x, self.weights[...], self.beta[...])
        elif self.method in ('godel', 'kleene_dienes'):
            return F.weighted_or_kleene_dienes(x, self.weights[...])
        elif self.method in ('product', 'reichenbach'):
            return F.weighted_or_reichenbach(x, self.weights[...])
        else:
            raise ValueError(f"Parametric OR method '{self.method}' is not supported.")


class WeightedAnd(nnx.Module):
    """
    Trainable parametric fuzzy AND gate supporting standard t-norms.

    This module implements a stateful intersection layer that combines multiple input 
    interval truth values into a single consolidated truth interval. It employs learnable
    weights to scale feature importance and a learnable beta parameter as an intensity bias.

    Attributes:
        method (str): Target logical framework selector ('lukasiewicz', 'godel', 
            'kleene_dienes', 'product', 'reichenbach').
        weights (nnx.Param): Trainable input importance weights structured as (num_inputs,).
        beta (nnx.Param): Trainable gate activation sensitivity threshold scalar (bias).
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the stateful WeightedAnd gate with corresponding optimization parameters.

        Args:
            num_inputs (int): Number of input streams to aggregate.
            rngs (nnx.Rngs): Flax NNX random number generator collection.
            method (str, optional): Fuzzy logic framework to employ. Defaults to 'lukasiewicz'.
        """
        self.method = method
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the parameterized forward pass for the fuzzy AND operation.

        Args:
            x (jnp.ndarray): Input interval tensor structured as (..., num_inputs, 2),
                where the final dimension contains the lower and upper bounds [L, U].

        Returns:
            jnp.ndarray: Evaluated collective truth interval structured as (..., 2).

        Raises:
            ValueError: If the specified `method` is not supported by the parametric AND layer.
        """
        if self.method == 'lukasiewicz':
            return F.weighted_and(x, self.weights[...], self.beta[...])
        elif self.method in ('godel', 'kleene_dienes'):
            return F.weighted_and_kleene_dienes(x, self.weights[...])
        elif self.method in ('product', 'reichenbach'):
            return F.weighted_and_reichenbach(x, self.weights[...])
        else:
            raise ValueError(f"Parametric AND method '{self.method}' is not supported.")


class WeightedNand(nnx.Module):
    """
    Trainable parametric fuzzy NAND gate.

    This module performs a parametric fuzzy conjunction (AND) across multiple 
    input interval channels, followed by a logical inversion (NOT) wrapper, using 
    trainable logic mechanics.

    Attributes:
        method (str): Target logical framework selector ('lukasiewicz', 'godel', 
            'kleene_dienes', 'product', 'reichenbach').
        weights (nnx.Param): Trainable input importance weights structured as (num_inputs,).
        beta (nnx.Param): Trainable gate activation sensitivity threshold scalar (bias).
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the stateful WeightedNand gate with corresponding optimization parameters.

        Args:
            num_inputs (int): Number of input streams to aggregate.
            rngs (nnx.Rngs): Flax NNX random number generator collection.
            method (str, optional): Fuzzy logic framework to employ. Defaults to 'lukasiewicz'.
        """
        self.method = method
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the parameterized forward pass for the fuzzy NAND operation.

        Args:
            x (jnp.ndarray): Input interval tensor structured as (..., num_inputs, 2),
                where the final dimension contains the lower and upper bounds [L, U].

        Returns:
            jnp.ndarray: Inverted collective truth interval structured as (..., 2).

        Raises:
            ValueError: If the specified `method` is not supported by the parametric NAND layer.
        """
        if self.method == 'lukasiewicz':
            return F.weighted_nand(x, self.weights[...], self.beta[...])
        elif self.method in ('godel', 'kleene_dienes'):
            return F.weighted_nand_kleene_dienes(x, self.weights[...])
        elif self.method in ('product', 'reichenbach'):
            return F.weighted_nand_reichenbach(x, self.weights[...])
        else:
            raise ValueError(f"Parametric NAND method '{self.method}' is not supported.")


class WeightedNor(nnx.Module):
    """
    Trainable parametric fuzzy NOR gate.

    This module performs a parametric fuzzy disjunction (OR) across multiple 
    input interval channels, followed by a logical inversion (NOT) wrapper, using 
    trainable logic mechanics.

    Attributes:
        method (str): Target logical framework selector ('lukasiewicz', 'godel', 
            'kleene_dienes', 'product', 'reichenbach').
        weights (nnx.Param): Trainable input importance weights structured as (num_inputs,).
        beta (nnx.Param): Trainable gate activation sensitivity threshold scalar (bias).
    """
    def __init__(self, num_inputs: int, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the stateful WeightedNor gate with corresponding optimization parameters.

        Args:
            num_inputs (int): Number of input streams to aggregate.
            rngs (nnx.Rngs): Flax NNX random number generator collection.
            method (str, optional): Fuzzy logic framework to employ. Defaults to 'lukasiewicz'.
        """
        self.method = method
        self.weights = nnx.Param(jnp.ones((num_inputs,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the parameterized forward pass for the fuzzy NOR operation.

        Args:
            x (jnp.ndarray): Input interval tensor structured as (..., num_inputs, 2),
                where the final dimension contains the lower and upper bounds [L, U].

        Returns:
            jnp.ndarray: Inverted collective truth interval structured as (..., 2).

        Raises:
            ValueError: If the specified `method` is not supported by the parametric NOR layer.
        """
        if self.method == 'lukasiewicz':
            return F.weighted_nor(x, self.weights[...], self.beta[...])
        elif self.method in ('godel', 'kleene_dienes'):
            return F.weighted_nor_kleene_dienes(x, self.weights[...])
        elif self.method in ('product', 'reichenbach'):
            return F.weighted_nor_reichenbach(x, self.weights[...])
        else:
            raise ValueError(f"Parametric NOR method '{self.method}' is not supported.")


class WeightedXor(nnx.Module):
    """
    Trainable parametric fuzzy XOR (Exclusive OR) gate.

    A binary trainable fuzzy logic layer that evaluates the strict difference or exclusive 
    disjunction between two distinct truth interval vectors under parameterized conditions.

    Attributes:
        method (str): Target logical framework selector ('lukasiewicz', 'godel', 
            'kleene_dienes', 'product', 'reichenbach').
        weights (nnx.Param): Trainable binary input interaction weights structured as (2,).
        beta (nnx.Param): Trainable activation sensitivity threshold scalar (bias).
    """
    def __init__(self, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the stateful WeightedXor gate with binary optimization attributes.

        Args:
            rngs (nnx.Rngs): Flax NNX random number generator collection.
            method (str, optional): Fuzzy logic framework to employ. Defaults to 'lukasiewicz'.
        """
        self.method = method
        self.weights = nnx.Param(jnp.ones((2,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the parameterized forward pass for the binary fuzzy XOR operation.

        Args:
            int_a (jnp.ndarray): First input interval tensor structured as (..., 2).
            int_b (jnp.ndarray): Second input interval tensor structured as (..., 2).

        Returns:
            jnp.ndarray: Evaluated XOR truth interval structured as (..., 2).

        Raises:
            ValueError: If the specified `method` is not supported by the parametric XOR layer.
        """
        if self.method == 'lukasiewicz':
            return F.weighted_xor_lukasiewicz(int_a, int_b, self.weights[...], self.beta[...])
        elif self.method in ('godel', 'kleene_dienes'):
            return F.weighted_xor_godel(int_a, int_b, self.weights[...])
        elif self.method in ('product', 'reichenbach'):
            return F.weighted_xor_product(int_a, int_b, self.weights[...])
        else:
            raise ValueError(f"Parametric XOR method '{self.method}' is not supported.")


class WeightedNot(nnx.Module):
    """
    Trainable parametric fuzzy NOT inversion gate.

    Applies a parameterized logic negation to an incoming interval tensor, allowing 
    the optimization routine to moderate or invert the intensity of the negative mapping via a learnable weight.

    Attributes:
        weight (nnx.Param): Trainable inversion scaling factor parameter.
    """
    def __init__(self, rngs: nnx.Rngs):
        """
        Initializes the stateful WeightedNot gate.

        Args:
            rngs (nnx.Rngs): Flax NNX random number generator collection.
        """
        self.weight = nnx.Param(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the parameterized fuzzy inversion pass.

        Args:
            x (jnp.ndarray): Input interval tensor structured as (..., 2).

        Returns:
            jnp.ndarray: Modulated inverted truth interval structured as (..., 2).
        """
        return F.weighted_not(x, self.weight[...])


class WeightedImplication(nnx.Module):
    """
    Trainable parametric implication gate (A -> B).

    Maps a causal rule structure where an antecedent interval tensor (A) implies 
    a consequent interval tensor (B). The gateway learns input relative scaling and standard bounds 
    corrections to preserve fuzzy logical constraints.

    Attributes:
        method (str): Target implication framework selector (e.g., 'lukasiewicz', 'reichenbach', 'kleene_dienes').
        weights (nnx.Param): Trainable rule component scaling weights structured as (2,).
        beta (nnx.Param): Trainable rule sensitivity threshold parameter (bias).
    """
    def __init__(self, rngs: nnx.Rngs, method: str = 'lukasiewicz'):
        """
        Initializes the stateful WeightedImplication layer with designated semantics.

        Args:
            rngs (nnx.Rngs): Flax NNX random number generator collection.
            method (str, optional): Implication model selection. Defaults to 'lukasiewicz'.
        """
        self.method = method
        self.weights = nnx.Param(jnp.ones((2,)))
        self.beta = nnx.Param(jnp.array(1.0))

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the parametric fuzzy rule implication forward pass.

        Args:
            int_a (jnp.ndarray): Antecedent truth interval tensor structured as (..., 2).
            int_b (jnp.ndarray): Consequent truth interval tensor structured as (..., 2).

        Returns:
            jnp.ndarray: Calculated rule validity interval tensor structured as (..., 2).
        """
        return F.weighted_implication(
            int_a, int_b, self.weights[...], self.beta[...], method=self.method
        )


# =====================================================================
# 2. PARAMETER-FREE PURE (BULK) REDUCTION GATES
# =====================================================================

class BulkAnd(nnx.Module):
    """
    Non-trainable pure stateless bulk AND reduction gate.

    Reduces a sequence of multiple truth intervals along the specified feature axis 
    into a single intersection interval. This is an unparameterized structural block 
    representing traditional multi-input fuzzy logic conjunction.

    Attributes:
        method (str): Pure fuzzy framework selector ('godel', 'kleene_dienes', 
            'product', 'reichenbach', 'lukasiewicz').
    """
    def __init__(self, method: str = 'kleene_dienes'):
        """
        Initializes the stateless BulkAnd reduction layer.

        Args:
            method (str, optional): Target fuzzy logic semantics. Defaults to 'kleene_dienes'.
        """
        self.method = method

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs structural bulk reduction using fuzzy AND axioms.

        Args:
            x (jnp.ndarray): Input multi-channel tensor structured as (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Consolidated reduction interval structured as (..., 2).

        Raises:
            ValueError: If the provided `method` is unknown or unhandled.
        """
        if self.method in ('godel', 'kleene_dienes'):
            return F.bulk_and_godel(x)
        elif self.method in ('product', 'reichenbach'):
            return F.bulk_and_product(x)
        elif self.method == 'lukasiewicz':
            inputs = [x[..., i, :] for i in range(x.shape[-2])]
            res = inputs[0]
            for item in inputs[1:]:
                res = F.and_lukasiewicz(res, item)
            return res
        else:
            raise ValueError(f"Bulk AND method '{self.method}' is not recognized.")


class BulkOr(nnx.Module):
    """
    Non-trainable pure stateless bulk OR reduction gate.

    Reduces a sequence of multiple truth intervals along the specified feature axis 
    into a single union interval. This is an unparameterized structural block 
    representing traditional multi-input fuzzy logic disjunction.

    Attributes:
        method (str): Pure fuzzy framework selector ('godel', 'kleene_dienes', 
            'product', 'reichenbach', 'lukasiewicz').
    """
    def __init__(self, method: str = 'kleene_dienes'):
        """
        Initializes the stateless BulkOr reduction layer.

        Args:
            method (str, optional): Target fuzzy logic semantics. Defaults to 'kleene_dienes'.
        """
        self.method = method

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs structural bulk reduction using fuzzy OR axioms.

        Args:
            x (jnp.ndarray): Input multi-channel tensor structured as (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Consolidated reduction interval structured as (..., 2).

        Raises:
            ValueError: If the provided `method` is unknown or unhandled.
        """
        if self.method in ('godel', 'kleene_dienes'):
            return F.bulk_or_godel(x)
        elif self.method in ('product', 'reichenbach'):
            return F.bulk_or_product(x)
        elif self.method == 'lukasiewicz':
            inputs = [x[..., i, :] for i in range(x.shape[-2])]
            res = inputs[0]
            for item in inputs[1:]:
                res = F.or_lukasiewicz(res, item)
            return res
        else:
            raise ValueError(f"Bulk OR method '{self.method}' is not recognized.")


# =====================================================================
# 3. SPECIAL PHYSICAL GATES (Physical Fuzzy Logic - PFL)
# =====================================================================

class PhysicalOr(nnx.Module):
    """
    Space-curved entropic physical OR gate with localized field configurations.

    Implements a field-theoretic physical fuzzy logic (PFL) disjunction. It projects 
    multi-input truth intervals into non-Euclidean space, evaluating field interactions 
    using explicit physical hyperparameters rather than learnable structural weights.

    Attributes:
        method (str): Target physical framework ('physical_godel', 'physical_kleene_dienes', 
            'physical_product', 'physical_reichenbach', 'physical_lukasiewicz').
        gamma (float): Space curvature dissipation coefficient parameter.
        mode (str): Activation field blending mode configuration (e.g., 'sigmoid').
        slope (float): Physical boundary transition slope parameter.
        offset (float): Physical potential zero-point offset.
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        """
        Initializes the parameter-free PhysicalOr field gate.

        Args:
            method (str, optional): Selected PFL operational semantics. Defaults to 'physical_kleene_dienes'.
            gamma (float, optional): Field interaction scaling. Defaults to 0.2.
            mode (str, optional): Wave/field mapping activation mode. Defaults to 'sigmoid'.
            slope (float, optional): Curvature transition coefficient. Defaults to 1.0.
            offset (float, optional): Geometric field center shift. Defaults to 0.5.
        """
        self.method = method
        self.gamma = gamma
        self.mode = mode
        self.slope = slope
        self.offset = offset

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes physical space-curved disjunction field dynamics.

        Args:
            x (jnp.ndarray): Input physical interval tensor structured as (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Synthesized output field interval tensor structured as (..., 2).

        Raises:
            ValueError: If the selected physical OR setup is not recognized.
        """
        inputs = [x[..., i, :] for i in range(x.shape[-2])]
        res = inputs[0]
        
        if self.method in ('physical_godel', 'physical_kleene_dienes'):
            for item in inputs[1:]:
                res = F.or_physical_kleene_dienes(res, item)
            return res
        elif self.method in ('physical_product', 'physical_reichenbach'):
            for item in inputs[1:]:
                res = F.or_physical_reichenbach(res, item)
            return res
        elif self.method == 'physical_lukasiewicz':
            for item in inputs[1:]:
                res = F.or_physical_lukasiewicz(res, item)
            return res
        else:
            raise ValueError(f"Physical OR method '{self.method}' is not supported.")


class PhysicalAnd(nnx.Module):
    """
    Space-curved entropic physical AND gate with localized field configurations.

    Implements a field-theoretic physical fuzzy logic (PFL) conjunction. It simulates 
    physical logic intersections across space-curved boundary zones using entropic 
    hyperparameters.

    Attributes:
        method (str): Target physical framework ('physical_godel', 'physical_kleene_dienes', 
            'physical_product', 'physical_reichenbach', 'physical_lukasiewicz').
        gamma (float): Space curvature dissipation coefficient parameter.
        mode (str): Activation field blending mode configuration (e.g., 'sigmoid').
        slope (float): Physical boundary transition slope parameter.
        offset (float): Physical potential zero-point offset.
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        """
        Initializes the parameter-free PhysicalAnd field gate.

        Args:
            method (str, optional): Selected PFL operational semantics. Defaults to 'physical_kleene_dienes'.
            gamma (float, optional): Field interaction scaling. Defaults to 0.2.
            mode (str, optional): Wave/field mapping activation mode. Defaults to 'sigmoid'.
            slope (float, optional): Curvature transition coefficient. Defaults to 1.0.
            offset (float, optional): Geometric field center shift. Defaults to 0.5.
        """
        self.method = method
        self.gamma = gamma
        self.mode = mode
        self.slope = slope
        self.offset = offset

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes physical space-curved conjunction field dynamics.

        Args:
            x (jnp.ndarray): Input physical interval tensor structured as (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Synthesized output field interval tensor structured as (..., 2).

        Raises:
            ValueError: If the selected physical AND setup is not recognized.
        """
        inputs = [x[..., i, :] for i in range(x.shape[-2])]
        res = inputs[0]
        
        if self.method in ('physical_godel', 'physical_kleene_dienes'):
            for item in inputs[1:]:
                res = F.and_physical_kleene_dienes(res, item)
            return res
        elif self.method in ('physical_product', 'physical_reichenbach'):
            for item in inputs[1:]:
                res = F.and_physical_reichenbach(res, item)
            return res
        elif self.method == 'physical_lukasiewicz':
            for item in inputs[1:]:
                res = F.and_physical_lukasiewicz(res, item)
            return res
        else:
            raise ValueError(f"Physical AND method '{self.method}' is not supported.")


class PhysicalImplication(nnx.Module):
    """
    Parameter-free space-curved rule gateway (A -> B).

    Evaluates rule mappings based on Physical Fuzzy Logic (PFL) metrics. It monitors 
    field potentials at logical boundary intersections (including continuous singularity zones)
    without utilizing learning states.

    Attributes:
        method (str): Selected physical implication kernel configuration.
        gamma (float): Space curvature dissipation coefficient parameter.
        mode (str): Activation field blending mode configuration.
        slope (float): Physical boundary transition slope parameter.
        offset (float): Physical potential zero-point offset.
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        """
        Initializes the non-parametric PhysicalImplication rule connector.

        Args:
            method (str, optional): Target PFL implication mapping. Defaults to 'physical_kleene_dienes'.
            gamma (float, optional): Field interaction scaling. Defaults to 0.2.
            mode (str, optional): Wave/field mapping activation mode. Defaults to 'sigmoid'.
            slope (float, optional): Curvature transition coefficient. Defaults to 1.0.
            offset (float, optional): Geometric field center shift. Defaults to 0.5.
        """
        self.method = method
        self.gamma = gamma
        self.mode = mode
        self.slope = slope
        self.offset = offset

    def __call__(self, int_a: jnp.ndarray, int_b: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the physical implication output based on causal field distributions.

        Args:
            int_a (jnp.ndarray): Physical antecedent field interval tensor structured as (..., 2).
            int_b (jnp.ndarray): Physical consequent field interval tensor structured as (..., 2).

        Returns:
            jnp.ndarray: Rules validity interval tensor structured as (..., 2).
        """
        return F.implication(int_a, int_b, method=self.method)


class PhysicalNot(nnx.Module):
    """
    Parameter-free physical inversion (NOT) gate.

    Provides a clean, static logic negation wrapper for structural logic field inversion 
    without any trainable parameter constraints.
    """
    def __init__(self):
        """Initializes the parameter-free PhysicalNot gate."""
        pass

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies mathematical logical inversion across the interval bounds.

        Args:
            x (jnp.ndarray): Input interval tensor structured as (..., 2).

        Returns:
            jnp.ndarray: Fully inverted truth interval tensor structured as (..., 2).
        """
        return F.logical_not(x)


class PhysicalNand(nnx.Module):
    """
    Parameter-free physical NAND gate.

    Composes a non-parametric field-theoretic physical AND operation followed by 
    a rigid physical inversion layer (NOT).

    Attributes:
        and_gate (PhysicalAnd): Internal physical intersection processor layer.
        not_gate (PhysicalNot): Internal physical logic negation layer.
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        """
        Initializes the compound PhysicalNand gate structure.

        Args:
            method (str, optional): Underlying PFL operational semantics. Defaults to 'physical_kleene_dienes'.
            gamma (float, optional): Field interaction scaling. Defaults to 0.2.
            mode (str, optional): Wave/field mapping activation mode. Defaults to 'sigmoid'.
            slope (float, optional): Curvature transition coefficient. Defaults to 1.0.
            offset (float, optional): Geometric field center shift. Defaults to 0.5.
        """
        self.and_gate = PhysicalAnd(method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
        self.not_gate = PhysicalNot()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes physical conjunction followed immediately by field negation.

        Args:
            x (jnp.ndarray): Input physical interval tensor structured as (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Evaluated physical NAND interval tensor structured as (..., 2).
        """
        and_res = self.and_gate(x)
        return self.not_gate(and_res)


class PhysicalNor(nnx.Module):
    """
    Parameter-free physical NOR gate.

    Composes a non-parametric field-theoretic physical OR operation followed by 
    a rigid physical inversion layer (NOT).

    Attributes:
        or_gate (PhysicalOr): Internal physical disjunction processor layer.
        not_gate (PhysicalNot): Internal physical logic negation layer.
    """
    def __init__(self, method: str = 'physical_kleene_dienes', gamma: float = 0.2, 
                 mode: str = 'sigmoid', slope: float = 1.0, offset: float = 0.5):
        """
        Initializes the compound PhysicalNor gate structure.

        Args:
            method (str, optional): Underlying PFL operational semantics. Defaults to 'physical_kleene_dienes'.
            gamma (float, optional): Field interaction scaling. Defaults to 0.2.
            mode (str, optional): Wave/field mapping activation mode. Defaults to 'sigmoid'.
            slope (float, optional): Curvature transition coefficient. Defaults to 1.0.
            offset (float, optional): Geometric field center shift. Defaults to 0.5.
        """
        self.or_gate = PhysicalOr(method=method, gamma=gamma, mode=mode, slope=slope, offset=offset)
        self.not_gate = PhysicalNot()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes physical disjunction followed immediately by field negation.

        Args:
            x (jnp.ndarray): Input physical interval tensor structured as (..., num_inputs, 2).

        Returns:
            jnp.ndarray: Evaluated physical NOR interval tensor structured as (..., 2).
        """
        or_res = self.or_gate(x)
        return self.not_gate(or_res)