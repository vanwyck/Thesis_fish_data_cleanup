# Thesis_fish_data_cleanup

**Note:** Run Cleanup_notebook for all results and explanation

**OndeDrive with all cleaned tracks** (only accesible with a UGent-account): https://ugentbe-my.sharepoint.com/:f:/g/personal/thomas_vanwyck_ugent_be/EvlcdGJBf2FEmnhhHDCwgIsBpF4ZfAY28HdTc-zspmAeUA?e=GtjILl

This repository provides all code for the implementation of the YAPS-BI algorithm, combined with a Kalman filter/smoother.

The method implemented here uses an altered version of the YAPS algorithm created by Baktoft (2017). 
This new version, YAPS-BI, uses the known mean of the burst interval and calculated soundspeeds to stabilize performance and speed up the process.

Additionally, the resulting YAPS track is filtered with a Kalman Filter and Rauch-Tung-Striebel smoother to reduce the impact of multipath errors further. 
Our results also show this additional step increases the relevance of the error measure significantly.

Results are compared to VPS tracks, cleaned up with a swimming speed filter and Kalman Filter.

This data clean-up system was developed and evaluated as part of the my Master's Disertation titled 'Comparison of data cleanup techniques for noisy fish positioning data'.
The example dataset is from European eel released in the Albert canal near Ham. This fish was tracked using the VEMCO positioning system. 
The data consists of random burst intervals and contains high levels of multipath errors. 

The development of this algorithm was possible thanks to the efforts of Jenna Vergeynst, Ingmar Nopens and the feedbak provided by Henrik Baktoft and James Campbell.