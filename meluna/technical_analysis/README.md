# Technical Analysis Library

A high-performance, streaming technical indicator library designed for event-driven backtesting and real-time trading applications.

---

## Features

* **O(1) Streaming Performance:** Efficient incremental calculations using rolling windows
* **Factory Pattern Interface:** Intuitive indicator creation with case-insensitive names
* **Composite Pattern Support:** Complex indicators built from simpler components
* **Thread-Safe Design:** Safe for multi-symbol processing
* **Event-Driven Integration:** Seamless integration with Meluna's backtesting framework

---

## Available Indicators

| Category | Indicators | Description |
|----------|------------|-------------|
| **Trend** | SMA, EMA | Moving averages for trend following |
| **Momentum** | RSI | Relative Strength Index |
| **Composite** | MACD, Bollinger Bands, Stochastic | Multi-component indicators |
| **Volatility** | RollingStdDev | Standard deviation calculations |
| **Min/Max** | RollingMinMax | Rolling minimum/maximum tracking |

---

## Core Architecture

* **BaseIndicator**: Abstract base class defining consistent interface
* **Factory**: Registry-based pattern for string-based creation
* **Indicators Module**: Concrete implementations organized by category
* **Exceptions**: Custom exception hierarchy for validation

## Performance

* **Memory**: O(period) constant usage per indicator
* **Updates**: O(1) streaming performance
* **Thread Safety**: Concurrent multi-symbol support