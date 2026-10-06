# uav-semantic-comm

Task-oriented generative semantic communication for UAV-to-ground image classification. ECE 627/527 (Fall 2026), Project Assignment 5, UMass Amherst.

Nikhil Venugopal, AJ El-Hout

Overview

A UAV sends images to a ground station over a bandwidth-limited wireless link. Since the ground station mostly needs the class label rather than an exact copy of the image, we send a compact task-relevant representation instead of full image data. The receiver uses a generative model to infer or rebuild useful content, and the system adapts how much it sends based on channel quality and mission priority.

We compare against a conventional compression + channel coding pipeline and a non-generative learned autoencoder (DeepJSCC-style) baseline.
