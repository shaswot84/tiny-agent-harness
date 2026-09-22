"""Collection of standard, ready-to-use tools and helper functions for TinyAgent."""

import math
import subprocess


def multiply(a: float, b: float) -> str:
    """Multiply two numbers together.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The product of a and b as a string.
    """
    return str(float(a) * float(b))


def add(a: float, b: float) -> str:
    """Add two numbers together.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of a and b as a string.
    """
    return str(float(a) + float(b))


def subtract(a: float, b: float) -> str:
    """Subtract b from a.

    Args:
        a: The first number.
        b: The second number to subtract.

    Returns:
        The difference as a string.
    """
    return str(float(a) - float(b))


def divide(a: float, b: float) -> str:
    """Divide a by b.

    Args:
        a: Numerator.
        b: Denominator (cannot be zero).

    Returns:
        The quotient as a string or an error message.
    """
    if float(b) == 0.0:
        return "Error: Division by zero."
    return str(float(a) / float(b))


def power(base: float, exponent: float) -> str:
    """Raise a number to a given power.

    Args:
        base: The base number.
        exponent: The exponent power.

    Returns:
        The result of base raised to exponent as a string.
    """
    return str(math.pow(float(base), float(exponent)))


def execute_command(command: str) -> str:
    """Execute a bash shell command and return its stdout/stderr.

    Note: This is a sensitive tool and should be configured with requires_approval.

    Args:
        command: The shell command line to run.

    Returns:
        Command output or error message.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout or result.stderr
        return output.strip() if output else "Command executed with no output."
    except Exception as e:
        return f"Error executing command: {e}"


def final_answer(answer: str) -> str:
    """Provide the final answer to the user and conclude tool usage.

    Args:
        answer: The final response to give to the user.

    Returns:
        The final answer text.
    """
    return answer
