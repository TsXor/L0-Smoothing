from .normal import L0_Smoothing as L0_Smoothing
from .pyocl import L0_Smoothing as L0_Smoothing_accel
from .pyocl import L0_Smoothing_CL as L0_Smoothing_CL

__all__ = ['L0_Smoothing', 'L0_Smoothing_accel', 'L0_Smoothing_CL']