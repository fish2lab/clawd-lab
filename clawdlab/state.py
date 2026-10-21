from dataclasses import dataclass, field

from .gfx import Cam


@dataclass
class State:
    t: float = 0.0
    cam: Cam = None
    cam_rot: float = 0.0
    shake: tuple = (0.0, 0.0)
    night: float = 0.0
    lamps: dict = field(default_factory=dict)
    lamp_sway: float = 0.0
    pods: dict = field(default_factory=dict)
    reactor: dict = field(default_factory=dict)
    chars: list = field(default_factory=list)
    bursts: list = field(default_factory=list)
    fx_back: list = field(default_factory=list)
    fx_front: list = field(default_factory=list)
    overlay: list = field(default_factory=list)
    alarm: float = 0.0
    flash: float = 0.0
    fade: float = 0.0
    black: bool = False
