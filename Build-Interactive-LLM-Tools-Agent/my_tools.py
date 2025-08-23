from langchain_core.tools import tool 


@tool
def add(a: int, b: int) -> int:
    """
    Add a and b.
    
    Args:
        a (int): first integer to be added
        b (int): second integer to be added

    Return:
        int: sum of a and b
    """
    return a + b


@tool
def subtract(a: int, b:int) -> int:
    """Subtract b from a."""
    return a - b

@tool
def multiply(a: int, b:int) -> int:
    """Multiply a and b."""
    return a * b


@tool 
def calculate_tip(total_bill: int, tip_percent: int) -> int: 
    ''' calculate tip'''
    return total_bill * tip_percent * 0.01


def __main__(): 
    inputs = {
        "total_bill": 120, 
        "tip_percent": 15
    }
    response = calculate_tip.invoke(inputs)
    print(response) 