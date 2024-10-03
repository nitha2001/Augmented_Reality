# Augmented_Reality

In this project, I implemented a simple augmented reality application that utilizes AprilTags for camera pose estimation. The objective is to recover camera poses through two methods: solving the Perspective-N-Point (PnP) problem and the Perspective-Three-Point (P3P) problem. The final deliverable includes a video showcasing virtual objects integrated into a real-world scene, demonstrating the practical application of computer vision techniques.

## Contents

1. est Pw.py
This function is responsible for finding world coordinates of a, b, c and d given
tag size
2. solve pnp.py
This function is responsible for recovering R and t from 2D-3D correspondence
with coplanar assumption
3. est pixel world.py
This function is responsible for solving 3D locations of a pixel on the table.
4. solve p3p.py
This file has two functions P3P and Procrustes that you need to write. P3P solves
the polynomial and calculates the distances of the 3 points from the camera and
then uses the corresponding coordinates in the camera frame and the world frame.
You need to call Procrustes inside P3P to return R,t.
5. VR res.gif

## Results
   ![image](https://github.com/user-attachments/assets/704bf49d-b6b2-496e-b90e-a8e515be75b9)

   ![image](https://github.com/user-attachments/assets/b7658ad8-2fa6-417e-8bbf-996b50d06b55)

