import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from drone_show import visualize

def test_visualize_uses_last_frame():
    """Verify that plot_frame uses X_final explicitly, not X[0]."""
    # Create X where X[0] != X[-1]
    N = 3
    X_start = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    X_final = np.array([[10.0, 10.0], [11.0, 10.0], [12.0, 10.0]])
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Call plot_frame with explicit X_start and X_final
    visualize.plot_frame(ax, X=None, targets=None, title="Test", 
                        bounds=(-1, 15, -1, 15), 
                        X_start=X_start, X_final=X_final)
    
    # Extract scatter plot data from axes
    # Find the scatter plot corresponding to "Drones (Final)"
    collections = ax.collections
    
    # Should have at least 2 scatter plots: Start and Final
    assert len(collections) >= 2, "Expected at least Start and Final scatter plots"
    
    # Find the final positions scatter plot
    # The last collection should be the final one (plotted last)
    # Or we can check by getting all scatter data
    final_found = False
    start_found = False
    
    for coll in collections:
        offsets = coll.get_offsets()
        if len(offsets) == N:
            # Check if it matches X_final
            if np.allclose(offsets, X_final, atol=0.1):
                final_found = True
            # Check if it matches X_start
            if np.allclose(offsets, X_start, atol=0.1):
                start_found = True
    
    assert start_found, "X_start positions not found in plot"
    assert final_found, "X_final positions not found in plot"
    
    # Verify X_final is actually at [10, 10] etc, not [0, 0]
    # This ensures we're using the explicit parameter, not accidentally using X[0]
    all_offsets = []
    for coll in collections:
        all_offsets.extend(coll.get_offsets())
    
    # Check that we have points near [10, 10] (X_final)
    has_final_positions = any(np.allclose(offset, [10.0, 10.0], atol=1.0) for offset in all_offsets)
    assert has_final_positions, "Final positions ([10,10] area) not found in plot"
    
    plt.close(fig)
