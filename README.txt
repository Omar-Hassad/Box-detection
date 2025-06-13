# Box-detection
Box detection system that measures length, width, and height using a calibrated Results can be exported automatically to Excel

1.In the file calibre.py, choose a box as your calibration reference. Enter the real length and width of this box, then run the code. It will generate two variables, a and b, which will be used by project.py.

2.Now run project.py. You can place your box anywhere, regardless of its position — the system will detect it. An Excel file will be generated, showing the measured results.

3.The height of the box is calculated separately, and the result is also added to the same Excel file.

4.Before any step you need to change the path of the best.onnx in all the file.py to ure oun path in your pc (you will find it in the top of the code)
