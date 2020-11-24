# image Compression 
Original codes of three methods<br>
Dataset in image Folder.(or download from http://netdissect.csail.mit.edu/data/broden1_227.zip)<br>
Videos in video Folder.

## SVD
Python 3.7.7<br>
Install all the necessary package and select the code segment want to run in main()

## K-mean Folder

### comparison for different Ks
Software:MATLAB<br>
1.download all files in MATLAB folder into the same folder<br> 2. Run PictureCompress file.<br>
output:pic(compressed.png) shows compressed images of 5 different K values  

### test for efficiency
Software:Python<br>
1.download all files in Python folder into the same folder.<br>
2.find the path of the folder contains 600 images. <br>
3.put the folder path replace the path parameter(k-mean.py-line 105)<br>
4.run it(It could take one day long)<br>
output:pics1-2 show running time of 5 K values.
