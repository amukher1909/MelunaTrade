"""Factory for creating technical indicators."""

import inspect
from typing import Dict, Any, Type, List, Union, Optional

from .base import BaseIndicator
from .exceptions import IndicatorError, InvalidParameterError, IndicatorNotFoundError
from .indicators.trend import SMA, EMA
from .indicators.momentum import RSI
from .indicators.composite import MACD, BollingerBands, Stochastic


class IndicatorRegistry:
    """Registry for managing indicators with aliases."""
    
    def __init__(self):
        """Initialize registry with built-in indicators."""
        self._registry: Dict[str, Type[BaseIndicator]] = {}
        self._register_builtin_indicators()
    
    def _register_builtin_indicators(self) -> None:
        """Register built-in indicators."""
        # Trend indicators
        self.register('sma', SMA, aliases=['simple_ma', 'simple_moving_average'])
        self.register('ema', EMA, aliases=['exp_ma', 'exponential_moving_average'])
        
        # Momentum indicators  
        self.register('rsi', RSI, aliases=['relative_strength_index'])
        
        # Composite indicators
        self.register('macd', MACD, aliases=['moving_average_convergence_divergence'])
        self.register('bollinger_bands', BollingerBands, aliases=['bbands', 'bb'])
        self.register('stochastic', Stochastic, aliases=['stoch'])
    
    def register(self, name: str, indicator_class: Type[BaseIndicator], aliases: Optional[List[str]] = None) -> None:
        """Register indicator with aliases."""
        name_lower = name.lower()
        self._registry[name_lower] = indicator_class
        
        if aliases:
            for alias in aliases:
                self._registry[alias.lower()] = indicator_class
    
    def get(self, name: str) -> Type[BaseIndicator]:
        """Get indicator class by name."""
        name_lower = name.lower()
        if name_lower not in self._registry:
            raise IndicatorNotFoundError(name, self.list_indicators())
        
        return self._registry[name_lower]
    
    def list_indicators(self) -> List[str]:
        """List available indicators."""
        # Get unique class names to avoid showing aliases
        unique_classes = set(self._registry.values())
        class_names = [cls.__name__.lower() for cls in unique_classes]
        return sorted(class_names)
    
    def get_aliases(self, name: str) -> List[str]:
        """
        Get all aliases for an indicator.
        
        Args:
            name (str): Indicator name
            
        Returns:
            List[str]: List of all names (including aliases) for the indicator
        """
        try:
            target_class = self.get(name)
            return [key for key, cls in self._registry.items() if cls == target_class]
        except IndicatorNotFoundError:
            return []


# Global registry instance
_REGISTRY = IndicatorRegistry()


def create(name: str, **kwargs) -> BaseIndicator:
    """
    Factory function to create technical indicators by name.
    
    This is the primary entry point for creating indicators. Provides
    case-insensitive indicator creation with comprehensive parameter validation.
    
    Args:
        name (str): Name of the indicator to create (case-insensitive).
            Available indicators can be listed using list_indicators().
        **kwargs: Parameters to pass to the indicator constructor.
            Common parameters:
            - period (int): Lookback period for calculations
            - input_field (str): Input field ('open', 'high', 'low', 'close', 'volume')
            - Additional parameters depend on the specific indicator
    
    Returns:
        BaseIndicator: Configured indicator instance ready for use
        
    Raises:
        IndicatorNotFoundError: If the indicator name is not recognized
        InvalidParameterError: If parameters are invalid or missing
        
    Examples:
        >>> import meluna.technical_analysis as ta
        >>> 
        >>> # Create basic indicators
        >>> sma = ta.create('sma', period=20)
        >>> ema = ta.create('ema', period=12, alpha=0.15)
        >>> rsi = ta.create('rsi', period=14)
        >>> 
        >>> # Create composite indicators
        >>> macd = ta.create('macd', fast_period=12, slow_period=26, signal_period=9)
        >>> bbands = ta.create('bollinger_bands', period=20, k=2.0)
        >>> stoch = ta.create('stochastic', k_period=14, d_period=3)
        >>> 
        >>> # Use different input fields
        >>> high_sma = ta.create('sma', period=10, input_field='high')
        >>> 
        >>> # Case-insensitive names work
        >>> macd2 = ta.create('MACD', fast_period=12, slow_period=26, signal_period=9)
    """
    try:
        indicator_class = _REGISTRY.get(name)
        return indicator_class(**kwargs)
    except TypeError as e:
        # Convert constructor errors to our custom exception
        sig = inspect.signature(indicator_class.__init__)
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        
        raise InvalidParameterError(
            parameter_name="constructor",
            value=str(kwargs),
            expected=f"valid parameters for {name}: {params}",
            indicator_name=name
        ) from e


def list_indicators() -> List[str]:
    """
    Get a list of all available indicator names.
    
    Returns:
        List[str]: Alphabetically sorted list of indicator names
        
    Example:
        >>> import meluna.technical_analysis as ta
        >>> indicators = ta.list_indicators()
        >>> print(indicators)
        ['bollinger_bands', 'ema', 'macd', 'rsi', 'sma', 'stochastic']
    """
    return _REGISTRY.list_indicators()


def describe(name: str) -> Dict[str, Any]:
    """
    Get detailed information about an indicator including parameters and documentation.
    
    Args:
        name (str): Name of the indicator to describe (case-insensitive)
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - name: Canonical class name
            - aliases: List of alternative names
            - parameters: Parameter information from constructor signature
            - docstring: Class documentation
            - required_inputs: Required input fields for the indicator
            
    Raises:
        IndicatorNotFoundError: If the indicator name is not recognized
        
    Example:
        >>> import meluna.technical_analysis as ta
        >>> info = ta.describe('macd')
        >>> print(info['name'])
        'MACD'
        >>> print(info['parameters'].keys())
        dict_keys(['fast_period', 'slow_period', 'signal_period', 'input_field'])
    """
    indicator_class = _REGISTRY.get(name)
    
    # Extract parameter information from constructor signature
    sig = inspect.signature(indicator_class.__init__)
    parameters = {}
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
            
        param_info = {
            'type': param.annotation if param.annotation != inspect.Parameter.empty else 'Any',
            'default': param.default if param.default != inspect.Parameter.empty else None,
            'required': param.default == inspect.Parameter.empty
        }
        parameters[param_name] = param_info
    
    return {
        'name': indicator_class.__name__,
        'aliases': _REGISTRY.get_aliases(name),
        'parameters': parameters,
        'docstring': indicator_class.__doc__,
        'required_inputs': getattr(indicator_class, 'required_inputs', ()),
    }


def validate_period(period: Any, name: str = "period") -> int:
    """
    Validate period parameter for indicators.
    
    Args:
        period (Any): The period value to validate
        name (str): Parameter name for error messages
        
    Returns:
        int: Validated period value
        
    Raises:
        InvalidParameterError: If period is invalid
    """
    if not isinstance(period, int):
        raise InvalidParameterError(name, period, "positive integer")
    
    if period <= 0:
        raise InvalidParameterError(name, period, "positive integer (> 0)")
    
    return period


def validate_alpha(alpha: Any) -> float:
    """
    Validate alpha parameter for EMA indicators.
    
    Args:
        alpha (Any): The alpha value to validate
        
    Returns:
        float: Validated alpha value
        
    Raises:
        InvalidParameterError: If alpha is invalid
    """
    if not isinstance(alpha, (int, float)):
        raise InvalidParameterError("alpha", alpha, "numeric value between 0 and 1")
    
    if not 0 < alpha <= 1:
        raise InvalidParameterError("alpha", alpha, "value between 0 and 1 (exclusive of 0)")
    
    return float(alpha)


def validate_k_factor(k: Any) -> float:
    """
    Validate k factor parameter for Bollinger Bands.
    
    Args:
        k (Any): The k factor value to validate
        
    Returns:
        float: Validated k factor value
        
    Raises:
        InvalidParameterError: If k factor is invalid
    """
    if not isinstance(k, (int, float)):
        raise InvalidParameterError("k", k, "positive numeric value")
    
    if k <= 0:
        raise InvalidParameterError("k", k, "positive value (> 0)")
    
    return float(k)


def validate_input_field(input_field: Any) -> str:
    """
    Validate input field parameter.
    
    Args:
        input_field (Any): The input field to validate
        
    Returns:
        str: Validated input field
        
    Raises:
        InvalidParameterError: If input field is invalid
    """
    if not isinstance(input_field, str):
        raise InvalidParameterError("input_field", input_field, "string")
    
    valid_fields = ['open', 'high', 'low', 'close', 'volume']
    if input_field.lower() not in valid_fields:
        raise InvalidParameterError(
            "input_field", 
            input_field, 
            f"one of {valid_fields}"
        )
    
    return input_field.lower()