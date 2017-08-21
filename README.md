# reporter-quality-testing-rig
OTv2: Docker containers and Jupyter Notebooks for testing Open Traffic Reporter quality using synthesized GPS data


### Setting up test environment
1. Download tiles (see [here](https://github.com/opentraffic/reporter/tree/dev/py))
2. Export env vars:
    - `export VALHALLA_DOCKER_DATAPATH=</path/to/valhalla/tiles.tar>`
3. Build reporter-qa image with `docker build -t opentraffic/reporter-quality-testing-rig:latest --force-rm .`
4. Export env var for route generation:
	- `export MAPZEN_API=<your_mapzen_api_key>`
5. Start the service: `docker-compose up`
6. Navigate to `localhost:8888` in a browser to explore the pre-configured Jupyter notebooks.
