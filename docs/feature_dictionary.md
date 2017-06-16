# Feature Dictionary

| Feature Name              | Description                                                       | Units     | 
|---------------------------|------------------------                                           |-------    |
| angular_rate_frenet       | Angular rate relative to curvature of the road                    | rad / s   |
| relative_offset           | lane offset, positive is to left                                  | m         |
| fore_fore_m_dist          | distance to vehicle in front of vehicle in front                  | m         |
| fore_m_is_avail           | 1 if no vehicle in front                                          | bool      |
| angular_rate_global       | turn rate from global perspective                                 | rad / s   |
| fore_r_vel                | velocity of vehicle in front and in lane to the right             | m / s     |
| is_colliding              | 1 if ego is currently colliding with another vehicle              | bool      |      
| jerk                      | Derivative of acceleration                                        | m / s^4   |
| lane_curvature            | global curvature of lane at ego position                          | rad       |           
| lane_offset_left          | distance to the left boundary of the lane                         | m         |
| lane_offset_left_is_avail | 1 if lane offset left is not available                            | bool      |
| length                    | vehicle length                                                    | m         |
| lidar_1                   | a particular lidar beam distance                                  | m         |
| time_to_collision         | time until a collision occurs if no acceleration occurs           | s         |
| rear_l_vel                | velocity of the rear left neighbor in the scene, if one exists    | m/s       |
| rangerate_lidar_6         | the rate of change of lidar number 6 (specifies which beam)       | m/s       |
| lidar_3                   | the distance to some object of lidar number 3                     | m         |
| timegap                   |                                                                   |           |
| time_to_collision_is_avail|                                                                   |           |
| relvelocity               |                                                                   |           |
| utilities                 |                                                                   |           |
