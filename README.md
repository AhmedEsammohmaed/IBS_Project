# IBS Project

This repository contains the standalone floor plan computer vision package in [`floorplan_cv_delivery/`](floorplan_cv_delivery/README.md), prepared as an integration component for the Private 5G Deployment Planner.

The GitHub repository was empty when this package was added. No existing FastAPI application or backend integration code was present to modify. The CV package exposes `floorplan_cv_delivery.src.predict_floorplan(image, options)` for integration by the application team.

The current pretrained checkpoint is wall-only. Additional classes and planner-specific evaluation require labeled data and separately trained/validated models; see the package README, model card, and evaluation report.
