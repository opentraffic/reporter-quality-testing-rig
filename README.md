# reporter-quality-testing-rig
OTv2: Docker containers and Jupyter Notebooks for testing Open Traffic Reporter quality using synthesized GPS data


### Setting up test environment
1. Download tiles (see [here](https://github.com/opentraffic/reporter/tree/dev/py))
2. Clone Valhalla docker [repo](https://github.com/valhalla/docker)
3. Build Valhalla docker image from ppa (this is slow):
    - `./build.sh ppa latest`
4. Clone `dev` branch of Open Traffic reporter [repo](https://github.com/opentraffic/reporter/tree/dev)
5. Build Open Traffic reporter image:
	- add the following parameter to the `RUN valhalla_build_config` line in the Dockerfile: `--meili-customizable mode,search_radius,turn_penalty_factor,gps_accuracy,sigma_z,beta`
	- `docker build -t opentraffic/reporter:latest --force-rm .`
6. Export env vars:
    - `export VALHALLA_DOCKER_DATAPATH=</path/to/valhalla/tiles.tar>`
7. Build reporter-qa image with `docker build -t opentraffic/reporter-quality-testing-rig:latest --force-rm .`
8. Export env var for route generation:
	- `export MAPZEN_API=<your_mapzen_api_key>`
9. Start the service: `docker-compose up`
10. Navigate to `localhost:8888` in a browser to explore the pre-configured Jupyter notebooks.


### TO DO:
- Steps 2-5 should soon be obsolete as soon as the latest changes to the Valhalla ppa have been pushed to dockerhub.