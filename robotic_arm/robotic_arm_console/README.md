# Robotic Arm Local Console

This interface displays the camera stream, object detections, commanded joint
angles, and pick-and-place task stages. It communicates with the vision and
control programs in the parent directory through a local Python gateway.

## Start the Console

Install dependencies and start the Python gateway:

```bash
npm install
npm run bridge
```

Open another terminal:

```bash
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000). The default gateway URL
is `http://127.0.0.1:8765`.

## Optional 3D Preview

The project-authored URDF is included at:

```text
public/robot/robotic_arm_v0_6.urdf
```

Download the printable STL files from
[Robotic Arm with Servo & Arduino](https://makerworld.com/en/models/1134925-robotic-arm-with-servo-arduino?from=search#profileId-1135927).
Place the STL files referenced by the URDF in `public/robot/` to enable the 3D
preview. The rest of the console remains available without the STL files.

## Operating Boundaries

- `Plan motion` generates a trajectory without sending physical commands.
- `Send motion` sends only the previously confirmed trajectory and requires an
  additional browser confirmation.
- The control payload is `J1,J2,J3,J4,J6`; J5 remains fixed.
- Physical zero positions, directions, and limits come from the Python control
  layer. The 3D model is for visualization only.
