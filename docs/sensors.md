## Mathematical Sensor Type Comparison

| Sensor Type | Mathematical Model | Measurement Vector | Real-World Examples |
|-------------|-------------------|-------------------|-------------------|
| **Bearing** (`brg`) | $z = \arctan2(y-y_s, x-x_s)$ | $z \in \mathbb{R}^1$ | Passive sonar, ESM, Direction finding |
| **Range** (`rng`) | $z = \sqrt{(x-x_s)^2 + (y-y_s)^2}$ | $z \in \mathbb{R}^1$ | Time-of-flight sensors, Laser rangefinder |
| **Bearing-Range** (`brg_rng`) | $\mathbf{z} = [\arctan2(y-y_s, x-x_s), \sqrt{(x-x_s)^2 + (y-y_s)^2}]^T$ | $\mathbf{z} \in \mathbb{R}^2$ | Radar, Active sonar, LiDAR |
| **Position** (`pos`) | $\mathbf{z} = [x, y]^T$ | $\mathbf{z} \in \mathbb{R}^2$ | GPS, Optical tracking, ADS-B |
| **3D Position** (`pos_3D`) | $\mathbf{z} = [x, y, z]^T$ | $\mathbf{z} \in \mathbb{R}^3$ | 3D LiDAR, Stereo vision, GNSS |
| **Doppler** (`brg_rr`) | $\mathbf{z} = [\arctan2(y-y_s, x-x_s), \frac{(x-x_s)\dot{x} + (y-y_s)\dot{y}}{\sqrt{(x-x_s)^2 + (y-y_s)^2}}]^T$ | $\mathbf{z} \in \mathbb{R}^2$ | Doppler radar, Police radar |
| **Military Radar** (`brg_rng_rngrt`) | $\mathbf{z} = [\theta, r, \dot{r}]^T$ | $\mathbf{z} \in \mathbb{R}^3$ | Military radar, Air traffic control |
| **3D LiDAR** (`az_el_rng`) | $\mathbf{z} = [\phi, \psi, r, \dot{r}]^T$ | $\mathbf{z} \in \mathbb{R}^4$ | 3D LiDAR, Phased array radar |

### Mathematical Notation
- $(x_s, y_s)$: Sensor position
- $\mathbf{x} = [x, \dot{x}, y, \dot{y}, \omega]^T$: Target state vector
- $\theta = \arctan2(y-y_s, x-x_s)$: Bearing angle
- $r = \sqrt{(x-x_s)^2 + (y-y_s)^2}$: Range
- $\dot{r} = \frac{(x-x_s)\dot{x} + (y-y_s)\dot{y}}{r}$: Range rate
- $\phi$: Azimuth angle, $\psi$: Elevation angle

## Configuration Examples

### 1. Bearing-Only Configuration (`configs/sensors/brg.yaml`)

```yaml
model: 'brg'
num_sensors: 6

sensors:
  positions:
    - [-2000, 0]      
    - [2000, 0]       
    - [2000, 2000]    
    - [-2000, 2000]   
    - [0, 2000]       
    - [0, 0]          

  bearing:
    type: 'brg'
    z_dim: 1
    noise_std: "0.2*np.pi/180"       
    detection_prob: [0.95, 0.95]      
    clutter_rate: [10, 10]           
    clutter_range: [0, "2*np.pi"]     
    pdf_c: "1/(2*np.pi)"              
```

### 2. Bearing-Range Configuration (`configs/sensors/brg_rng.yaml`)

```yaml
model: 'brg_rng'
num_sensors: 4

sensors:
  positions: [[-2000, 0], [2000, 0], [2000, 2000], [-2000, 2000]]
  velocities: [[0, 0], [0, 0], [0, -10], [10, 0]] 
  
  bearing_range:
    type: 'brg_rng'
    z_dim: 2
    noise_std: ["2*np.pi/180", 10]    
    detection_prob: [0.95, 0.95]     
    clutter_rate: [10, 10]            
    pdf_c: "1/(np.pi * 4000)"         
    
  clutter_ranges: 
    - [["-(np.pi/2)", "np.pi/2"], [0, 4000]]      
    - [["np.pi/2", "3*np.pi/2"], [0, 4000]]       
    - [["np.pi/2", "3*np.pi/2"], [0, 4000]]       
    - [["-(np.pi/2)", "np.pi/2"], [0, 4000]]      
```

### 3. Mixed Sensor Configuration (`configs/sensors/mixed.yaml`)

```yaml
model: 'mixed'
num_sensors: 6

sensors:
  sensor_configs:
    - type: 'brg'
      position: [-2000, 0]
      z_dim: 1
      noise_std: "0.2*np.pi/180"      
      detection_prob: [0.95, 0.95]
      clutter_rate: [10, 10]
      clutter_range: [0, "2*np.pi"]
      pdf_c: "1/(2*np.pi)"
      
    - type: 'brg'
      position: [2000, 0]
      z_dim: 1
      noise_std: "0.2*np.pi/180"
      detection_prob: [0.95, 0.95]
      clutter_rate: [10, 10]
      clutter_range: [0, "2*np.pi"]
      pdf_c: "1/(2*np.pi)"
      
    - type: 'brg'
      position: [2000, 2000]
      velocity: [0, -10]              
      z_dim: 1
      noise_std: "0.2*np.pi/180"
      detection_prob: [0.95, 0.95]
      clutter_rate: [10, 10]
      clutter_range: [0, "2*np.pi"]
      pdf_c: "1/(2*np.pi)"
      
    - type: 'brg_rng'
      position: [2000, 2000]
      velocity: [0, -10]
      z_dim: 2
      noise_std: ["2*np.pi/180", 10]  
      detection_prob: [0.95, 0.95]
      clutter_rate: [10, 10]
      clutter_range: [["np.pi/2", "3*np.pi/2"], [0, 4000]]
      pdf_c: "1/(np.pi * 4000)"
      
    - type: 'brg_rng'
      position: [-2000, 2000]
      velocity: [10, 0]
      z_dim: 2
      noise_std: ["2*np.pi/180", 10]
      detection_prob: [0.95, 0.95]
      clutter_rate: [10, 10]
      clutter_range: [["-(np.pi/2)", "np.pi/2"], [0, 4000]]
      pdf_c: "1/(np.pi * 4000)"
```