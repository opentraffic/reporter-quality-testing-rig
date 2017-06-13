# reporter-quality-testing-rig
OTv2: Docker containers and Jupyter Notebooks for testing Open Traffic Reporter quality using synthesized GPS data


### Setting up test environment
1. Clone Valhalla docker [repo](https://github.com/valhalla/docker)
2. Build Valhalla docker image from ppa (this is slow):
    - `./build.sh ppa latest`
3. Clone `dev` branch of Open Traffic reporter [repo](https://github.com/opentraffic/reporter/tree/kk_kafka)
4. Build Open Traffic reporter image:
	- `docker build -t opentraffic/reporter:latest --force-rm .`
5. Export env vars:
    - `export VALHALLA_DOCKER_DATAPATH=</path/to/valhalla/tiles.tar>`
    - `export DATAPATH=</path/to/opentraffic/reporter/tiles.tar>`
6. Build reporter-qa image with `docker build -t opentraffic/reporter-quality-testing-rig:latest --force-rm .`
7. Export env var for route generation:
	- `export MAPZEN_API=<your_mapzen_api_key>`
8. Start the service: `docker-compose up`
9. Navigate to `localhost:8888` in a browser to explore the pre-configured Jupyter notebooks.

### TODO
- resample fake GPS traces at user-defined rate
- enumerate params to tune:
    - sigma-z
    - beta
    - sampling rate
- expose params as top-level container args
	- currently can be set in valhalla Dockerfile