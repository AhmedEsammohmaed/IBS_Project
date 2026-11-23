# IBS Project

This repository contains the standalone floor plan computer vision package in [`floorplan_cv_delivery/`](floorplan_cv_delivery/README.md), prepared as an integration component for the Private 5G Deployment Planner.

The GitHub repository was empty when this package was added. No existing FastAPI application or backend integration code was present to modify. The CV package exposes `floorplan_cv_delivery.src.predict_floorplan(image, options)` for integration by the application team.

The current optional pretrained checkpoint covers multiple building-element classes and is licensed CC BY-NC 4.0. Commercial use requires separate permission. The package includes a drop-in FastAPI router, but the planner backend was not present in this initially empty repository. See the package README, model card, and evaluation report for verified scope and remaining deployment checks.
