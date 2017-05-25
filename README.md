# reporter-quality-testing-rig
OTv2: Docker containers and Jupyter Notebooks for testing Open Traffic Reporter quality using synthesized GPS data


### Setting up test environment
1. Get some tiles!
2. Export env vars (these can reference the same file):
    - `export VALHALLA_DOCKER_DATAPATH=</path/to/valhalla/tiles.tar>`
    - `export DATAPATH=</path/to/opentraffic/reporter/tiles.tar>`
3. Navigate to directory containing **docker-compose.yml**
4. Start the service: `docker-compose up`

### TODO
- start scoring the traffic matches
- enumerate params to tune:
    - sigma-z
    - beta
    - sampling rate
- expose params as top-level container args
