# Liveness Test System
## Introduction
This system is a facial recognition system designed for the web!

Built using a microservices architecture, relying on Docker, DockerCompose and GRPC.

## Folder Structure
Each folder within the app directory is a single microservice.

Each microservice has a single role within our system.

The application itself is managed through a central service called 'main'.

### Main
Main works with the other services (as specified in it's configuration file),
and is the central consolodation layer for all checks relating to the auth system.

It is fully exposed to the public web, to provide the user with a central portal
for the authentication layer.


