Hey I have this sleep detection code using mediapipe. I want to convert it into a backend api.

There should be three endpoints.

Auth Endpoint

An endpoint which takes in a series of 24 images and then if more than 12 frames are detected as drowsy return an alert message.

Third one which takes in realtime streaming data and then if continuosly for 24 frames eyes are detected as drowsy sent an alert.

I am attaching a sample copy of backend and frontend I have right now which detects face. Follow similar endpoint names and request and response structure but modified for drowsiness detection instead of face detection.

The current frontend code doesnt have realtime streaming code. Please add that functionality as well. Also add a auth page before all features are presented to the user. taking in name and password for now. And calls the auth endpoint and authenticates it with a hashcode which should be added to headers in all subsequent calls to other apis for validation.

Also add proper print statements for logging and add a TAG in the format "SMARTROADSAI | FileName | FeatureName" so that its easy to detect in grafana.

Also add "24" frames count as an environment variable so that I can configure it in my docker instead if I want to change it while running.

