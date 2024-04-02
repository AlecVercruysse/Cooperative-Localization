# Cooperative-Localization
###  Arya Goutam, Alec Vercruysse, Jonathan Lo

This is a final project for HMC's ENGR205: State Estimation, completed in Fall 2022.

This project aims to study the problems of cooperative localization. Cooperative localization is when multiple robots work together to find out where they are. This project implements simultaneous localization and mapping, but in the experiments with robot cooperation, only cooperative localization is considered.


Cooperative localization is useful when many robots are working together, such as in the case of swarm robotics. Swarm robotics are when multiple low-cost robots are used to solve a task instead of a single expensive robot. Swarm robotics have found various applications including tasks like distributed sensing, search and rescue, and distributed 3D printing. Cooperative localization is particularly useful in applications of swarm robotics because a crucial limiting factor to the size of the swarm is the cost of individual robots. If there exist localization modalities where only a few robots in the swarm need to have access to advanced localization sensors, this cost can be reduced. Cooperative localization is helpful to improve performance in any case where a fleet of robots is used, however.

If you're interested in reading more, you can find
 - our project report website [online](https://sites.google.com/g.hmc.edu/e205-final-project-site/home), and
 - an interactive visualization of both the Kalman filter and the Covariance Intersection algorithm [here](https://alecvercruysse.github.io/tools/sensor-fusion/visualize.html)
