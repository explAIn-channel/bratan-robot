# Third-party notices

## Adafruit M4 Eyes

`face/app/src/main/res/drawable-nodpi/m4_hazel_iris.png`, `m4_hazel_sclera.png` are adapted from the Hazel eye preset of
[Adafruit M4 Eyes](https://github.com/adafruit/Adafruit_Learning_System_Guides/tree/main/M4_Eyes).

```
MIT License

Copyright (c) 2019 Adafruit Industries

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## CRL-200S protocol

`board/body.c` implements the GD32 and Delta-2D wire formats independently, after checking them against the public
documentation in [VacuumTiger / sangam-io](https://github.com/codetiger/VacuumTiger). No code was copied.

## Models (not included)

The neural network models listed in the main README are downloaded separately and keep their own licenses
(Ultralytics YOLO, OpenCV Zoo YuNet/SFace, ONNX Model Zoo FER+, Silero VAD, Vosk, YAMNet, Piper voices).

## Lamp protocol

`face/.../MoodLighting.java` speaks the Zengge/Telink mesh protocol of off-the-shelf BLE lamps and uses the
manufacturer's public default mesh name and password, as the stock app does.
