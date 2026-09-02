import numpy as np
from cdts.metrics import extract_events
from cdts.ccdc import run_ccdc

def test_extract_events():
    # Shape: (max_vertices * 2, rows, cols)
    # Let's say max_segments = 2, max_vertices = 3
    stack = np.zeros((6, 2, 2), dtype=np.float32)
    
    # Pixel 0,0: Year=[2000, 2005, 2010], Value=[0.8, 0.2, 0.9]
    # Loss event: 2000 to 2005. Mag = 0.6, dur = 5, rate = 0.12
    # Gain event: 2005 to 2010. Mag = 0.7, dur = 5, rate = 0.14
    stack[0, 0, 0] = 2000; stack[3, 0, 0] = 0.8
    stack[1, 0, 0] = 2005; stack[4, 0, 0] = 0.2
    stack[2, 0, 0] = 2010; stack[5, 0, 0] = 0.9
    
    # Test loss extraction
    loss_events = extract_events(stack, event_type="loss", sort_by="greatest")
    assert loss_events["yod"][0, 0] == 2000
    assert np.isclose(loss_events["magnitude"][0, 0], 0.6)
    assert loss_events["duration"][0, 0] == 5
    
    # Test gain extraction
    gain_events = extract_events(stack, event_type="gain", sort_by="greatest")
    assert gain_events["yod"][0, 0] == 2005
    assert np.isclose(gain_events["magnitude"][0, 0], 0.7)
    
def test_ccdc_basic():
    # Generate dummy data for a stable period (2 years)
    dates = np.arange(1, 800, 16) # Landsat roughly every 16 days
    values = 100.0 + 50.0 * np.cos(2 * np.pi * dates / 365.25)
    
    # Introduce a massive break at index 25
    values[25:] -= 80.0
    
    qa = np.zeros(len(dates), dtype=int)
    
    # Run CCDC
    segments = run_ccdc(dates, values, qa)
    
    # It should have found 2 segments due to the break
    assert len(segments) >= 2
    assert segments[0]["t_break"] > 0
    # coefs is now a list of lists: coefs[band][param]
    assert len(segments[0]["coefs"]) == 1 # 1 band
    assert len(segments[0]["coefs"][0]) == 6 # 6 parameters
