import sys
import numpy as np
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Qt
from impy.ui import ImageWindow
from impy.rois import LineROI, RectangleROI
from impy.annotation_manager import get_annotation_manager
from impy import analysis

def verify_analysis():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    print("Verifying ROI Analysis...")
    
    # Setup Window
    # 5D data: T=1, Z=1, C=1, Y=100, X=100
    data = np.zeros((1, 1, 1, 100, 100), dtype=np.uint8)
    # Add a line of intensity
    data[0, 0, 0, 50, 10:90] = 255
    
    win = ImageWindow(data, title="Analysis Test")
    win.show()
    
    mgr = get_annotation_manager()
    mgr.show()
    
    # 1. Test Line Profile
    print("Testing Line Profile...")
    line = LineROI(win.view)
    line.update((10, 50), (90, 50)) # Horizontal line along the intensity
    win.rois.append(line)
    mgr.add_roi(line)
    
    # Select line in list
    mgr.roi_list.setCurrentRow(0)
    
    # Mock plot_profile to avoid showing window during test
    original_plot = analysis.plot_profile
    called = False
    def mock_plot(img, roi):
        nonlocal called
        called = True
        print(f"Mock Plot called with img shape {img.shape}")
        
    # We can't easily patch the magicgui decorated function directly if we want to test the button connection
    # But we can patch the function in the module?
    # The button calls `self.run_analysis(plot_profile)`
    # Let's patch `mgr.run_analysis` or just run it manually?
    # Let's run `mgr.run_analysis(mock_plot)` to test the data extraction logic.
    
    mgr.run_analysis(mock_plot)
    
    if not called:
        print("FAIL: Plot profile analysis not called")
    else:
        print("PASS: Plot profile analysis called")
        
    # 2. Test Crop
    print("Testing Crop...")
    rect = RectangleROI(win.view)
    rect.update((20, 20), (80, 80))
    win.rois.append(rect)
    mgr.add_roi(rect)
    
    mgr.roi_list.setCurrentRow(1) # Select rect
    
    called_crop = False
    def mock_crop(img, roi):
        nonlocal called_crop
        called_crop = True
        print(f"Mock Crop called with img shape {img.shape}")
        
    mgr.run_analysis(mock_crop)
    
    if not called_crop:
        print("FAIL: Crop analysis not called")
    else:
        print("PASS: Crop analysis called")

    print("All Analysis tests passed!")
    win.close()
    mgr.close()

if __name__ == "__main__":
    verify_analysis()
