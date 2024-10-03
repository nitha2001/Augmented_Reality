# Augmented_Reality

In this project, I implemented a simple augmented reality application that utilizes AprilTags for camera pose estimation. The objective is to recover camera poses through two methods: solving the Perspective-N-Point (PnP) problem and the Perspective-Three-Point (P3P) problem. The final deliverable includes a video showcasing virtual objects integrated into a real-world scene, demonstrating the practical application of computer vision techniques.

## Contents
- `est Pw.py`: This function is responsible for finding world coordinates of a, b, c and d given tag size.
- `solve pnp.py`: This function is responsible for recovering R and t from 2D-3D correspondence with coplanar assumption.
- `est pixel world.py`: This function is responsible for solving 3D locations of a pixel on the table.
- `solve p3p.py`:This file has two functions P3P and Procrustes that you need to write. P3P solves the polynomial and calculates the distances of the 3 points from the camera and then uses the corresponding coordinates in the camera frame and the world frame.
- `VR res.gif`: result gif

## Results
   ![image](https://github.com/user-attachments/assets/704bf49d-b6b2-496e-b90e-a8e515be75b9)

   ![image](https://github.com/user-attachments/assets/b7658ad8-2fa6-417e-8bbf-996b50d06b55)

![Alt Text]([https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif](https://drive.google.com/file/d/1mysj0NCkRuvQVaXLzdIwukcRm9eFshM3/view?usp=sharing))


