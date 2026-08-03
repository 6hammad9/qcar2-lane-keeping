"""CSV logging for lane detection."""

from typing import Optional


class LaneLogger:
    """Logs lane detection data to CSV."""
    
    def __init__(self, filepath: str):
        self._file = open(filepath, 'w')
        self._file.write(
            'frame,'
            'cl_top_x,cl_mid_x,cl_bot_x,'
            'rl_top_x,rl_mid_x,rl_bot_x,'
            'lane_centre_x,error\n'
        )
    
    def log(
        self,
        frame_num: int,
        cl_top_x: Optional[float],
        cl_mid_x: Optional[float],
        cl_bot_x: Optional[float],
        rl_top_x: Optional[float],
        rl_mid_x: Optional[float],
        rl_bot_x: Optional[float],
        lane_centre_x: Optional[float],
        error: float
    ):
        def v(x):
            return f'{x:.2f}' if x is not None else ''
        
        self._file.write(
            f'{frame_num},'
            f'{v(cl_top_x)},{v(cl_mid_x)},{v(cl_bot_x)},'
            f'{v(rl_top_x)},{v(rl_mid_x)},{v(rl_bot_x)},'
            f'{v(lane_centre_x)},{error:.4f}\n'
        )
        self._file.flush()
    
    def close(self):
        self._file.close()
