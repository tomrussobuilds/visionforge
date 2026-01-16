"""
Execution & Optimization Policies.

Defines decision-making logic for runtime strategy based on 
hardware availability and configuration constraints.
"""

def determine_tta_mode(use_tta: bool, device_type: str) -> str:
    """
    Defines TTA complexity based on hardware acceleration availability.
    
    Args:
        use_tta (bool): Whether Test-Time Augmentation is enabled.
        device_type (str): The type of active device ('cpu', 'cuda', 'mps').

    Returns:
        str: Descriptive string of the TTA operation mode.
    """
    if not use_tta:
        return "DISABLED"

    # CPU-based TTA can be extremely slow for large ensembles; 
    # we enforce a LIGHT policy unless acceleration is present.
    if device_type == "cpu":
        return "LIGHT (CPU Optimized)"
    
    return f"FULL ({device_type.upper()})"