In order to achieve accurate classification from the Akida model, we need to train it on realistic images, thus giving it simulated weather conditions.

For the first iteration I added basic filters like gaussian noise, motion blur, defocus blur, and jitter.
Reference image:
![[Satellite Image.png]]


**1. Gaussian Noise**
Gaussian Noise can be experienced because of the electronic gain. Bumping up the camera's sensor gain physically introduces electrica static (grain) into the image. Extreme cold (- 60C in the stratosphere) can also cause battery voltage drops that increase sensor noise.

![[Gaussian Noise Image.png]]


**2. Motion Blur**
If the payload box is spinning or swinging on its tether in high stratospheric winds, the camera is moving laterally. This is compensated by the `SHUTTER_US` variable in the camera code. which freezes most of the motion

![[Motion Blur Image.png]]


**3. Defocus Blur**
The camera lens is going to experience temperatures varying between +30C on the launchpad to -60C in the tropopause. Lenses mathematically shift their focal plane as the glass and plastic barrels contract in the cold so we are risking micro-condensation or frost forming on the outside of the payload lens casing.

![[Defocus Blur Image.png]]


**5. Jitter (Brightness & Contrast)**
Without a thich atmosphere above the satellite to scatter and soften the sun rays, direct sunlight bouncing of a lake, a tin roof, or a cloud layer will overload the camera sensor (overexposure). Conversely, if the payload swings and the camera points into the shadow of the balloon itself, the image will instantly darken (underexposure).

![[Jitter Image 1.png]]![[Jitter Image 2.png]]

Now I will be trying to simulate realistic conditions like sun glare, cloud coverage, etc.

**6. Rayleigh Scattering**
Looking down through 20-30 kilometers of atmosphere washes out terrain colors and significantly flattens contrast.

![[Rayleigh Image.png]]

**7. Sun Glare (Lens Flare)**
Because there is less atmosphere to scatter the sun light, rays hitting the camera lens creates harsh, directional flares and massive overexposure.

![[Sun Glare Image.png]]


**8. Cloud Coverage**
Spatial masking, simulating bright white patches physically blocking the terrain.
![[High Cloud Coverage.png]]![[Medium Cloud Coverage.png]]