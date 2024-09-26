
import os
import inspect

def caller_info(back_frames=1):
    back_frames += 1
    frame = inspect.currentframe()
    while back_frames > 0:
        frame = frame.f_back
        back_frames -= 1
    (filename, line_number, function_name, lines, index) = inspect.getframeinfo(frame)
    return "%s:%d %s" % (filename, line_number, function_name)
